"""部分決定係数（partial R²）による感度分析（Cinelli & Hazlett, 2020。R の sensemakr と同じ式）。

DML の部分線形モデル Y = θ·T + g(X, W) + ε では、nuisance の残差どうしの回帰
Ỹ = θ·T̃ + e（Ỹ = Y − E[Y|X,W]、T̃ = T − E[T|X,W]）で θ を求める。これは観測された
共変量を（機械学習で非線形に）取り除いた後の単回帰なので、OLS 向けの Cinelli & Hazlett の
枠組みをそのまま使える
（Chernozhukov et al. 2022 は、部分線形モデルで同じ偏りの式が成り立つことを示している）。

未観測の交絡因子 Z の強さを、次の2つの部分決定係数で表す。

- R²_{T~Z|X}（`r2dz`）: Z が、観測された共変量で説明できない処置の分散を説明する割合
- R²_{Y~Z|T,X}（`r2yz`）: Z が、処置と観測された共変量で説明できない結果の分散を説明する割合

このとき偏りの大きさは |bias| = se·√dof·√(r2yz·r2dz / (1 − r2dz)) で、調整後の推定値は推定値を
0 に近づける向きにずらしたもの。分かること:

- R²_{Y~T|X}: 処置が結果の残差を説明する割合。どんなに強い交絡でも、r2yz = 1 なら
  r2dz がこの値以上でないと推定値を 0 にできない（極端なシナリオの限界）
- robustness value RV_q: r2dz = r2yz としたとき、推定値を (1 − q) 倍にする（q = 1 なら 0 にする）
  交絡の強さ。RV_q,α は、CI が (1 − q) 倍の値を含むようになる強さ
- benchmark: 「観測された共変量 X_j の k 倍の強さの交絡」なら推定値がどうなるか
  （`benchmark_bounds`。強さの目安を観測された変数で与える）

econml の omitted variable bias 感度分析（`sensitivity.ovb_summary`）の c_y・c_t も同じ
部分決定係数で、因果フォレストの ATE を対象にしている。このモジュールは、残差回帰の推定値
（処置の残差の分散で重み付けした平均的な効果。CATE の単純平均とは少し違う）を対象に、
自由度を考えた RV と、観測された共変量による benchmark を求める。

注意:
- 自由度は dof = n − 1（残差どうしの単回帰）とする。nuisance を機械学習で推定しているので、
  OLS のように共変量の数だけ自由度が減るわけではない。標本が大きければ結果はほとんど変わらない。
- 標準誤差は OLS の古典的な式（不均一分散を考えない）。sensemakr の式と整合させるため。
- benchmark の部分決定係数は、X_j を除いて nuisance を学習し直したときの残差の分散の減り方から求める
  （非線形の nuisance での近似）。benchmark の式は線形モデルで導かれたもので、目安として使う。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator

from causal_inference.nuisance import Splitter, check_finite, cross_fit_predict

type ArrayLike = np.ndarray | pd.Series | float


@dataclass(frozen=True)
class DmlResiduals:
    """nuisance の cross-fitting の残差。

    Attributes:
        y_res: Y − E[Y|X,W] の out-of-fold の残差。
        t_res: T − E[T|X,W] の out-of-fold の残差。
    """

    y_res: np.ndarray
    t_res: np.ndarray


@dataclass(frozen=True)
class ResidualRegression:
    """残差どうしの単回帰 Ỹ = θ·T̃ + e の結果。

    Attributes:
        estimate: θ の推定値。
        std_error: θ の標準誤差（古典的な式）。
        dof: 自由度（n − 1）。
        residual_sd_y: e の標準偏差（Y を処置と観測された共変量で説明した残り）。
        residual_sd_t: T̃ の標準偏差（T を観測された共変量で説明した残り）。
    """

    estimate: float
    std_error: float
    dof: int
    residual_sd_y: float
    residual_sd_t: float

    @property
    def t_value(self) -> float:
        """t 値（推定値 / 標準誤差）。"""
        return self.estimate / self.std_error

    @property
    def partial_r2_yd(self) -> float:
        """R²_{Y~T|X}: 処置が結果の残差を説明する割合（t² / (t² + dof)）。"""
        t2 = self.t_value**2
        return t2 / (t2 + self.dof)

    def rescaled(self, factor: float) -> ResidualRegression:
        """効果の単位を `factor` 倍した結果を返す（部分決定係数・RV・t 値の大きさは変わらない）。

        Y を定数 c 倍して推定したときは `factor=1/c` で元の単位に戻し、T を T0 から T1 に
        変えたときの効果にするときは `factor=T1 − T0` を掛ける
        （負なら推定値・t 値の符号が反転する）。

        Args:
            factor: 倍率（0 以外の有限な値）。

        Returns:
            倍率を掛けた結果。

        Raises:
            ValueError: `factor` が 0 または有限でない場合。
        """
        if not (np.isfinite(factor) and factor != 0):
            raise ValueError(f"factor は 0 以外の有限な値にしてください: {factor}")
        return replace(
            self,
            estimate=self.estimate * factor,
            std_error=self.std_error * abs(factor),
            residual_sd_y=self.residual_sd_y * abs(factor),
        )


def residualize(
    y: np.ndarray | pd.Series,
    t: np.ndarray | pd.Series,
    controls: np.ndarray | pd.DataFrame,
    *,
    model_y: BaseEstimator,
    model_t: BaseEstimator,
    cv: Splitter,
) -> DmlResiduals:
    """Y・T を観測された共変量で cross-fitting して、out-of-fold の残差を返す。

    Args:
        y: 結果変数。
        t: 処置変数。
        controls: 観測された共変量（`[X, W]`）。
        model_y: E[Y|X,W] のモデル（未学習。fold ごとに clone する）。
        model_t: E[T|X,W] のモデル（未学習）。
        cv: cross-fitting の分割器。

    Returns:
        残差。

    Raises:
        ValueError: 欠損・inf を含む、または行数が合わない場合。
    """
    for name, values in (("Y", y), ("T", t), ("共変量", controls)):
        check_finite(name, values)
    y_arr = np.asarray(y, dtype=float).ravel()
    t_arr = np.asarray(t, dtype=float).ravel()
    features = np.asarray(controls, dtype=float)
    if not len(y_arr) == len(t_arr) == len(features):
        raise ValueError(
            f"行数が合いません: Y={len(y_arr)}, T={len(t_arr)}, 共変量={len(features)}"
        )
    y_res = y_arr - cross_fit_predict(model_y, features, y_arr, cv)
    t_res = t_arr - cross_fit_predict(model_t, features, t_arr, cv)
    return DmlResiduals(y_res=y_res, t_res=t_res)


def fit_residual_regression(residuals: DmlResiduals) -> ResidualRegression:
    """残差どうしの単回帰（切片なし）で θ を求める。

    Args:
        residuals: `residualize` の結果。

    Returns:
        回帰の結果。

    Raises:
        ValueError: 行数が 3 未満、または処置の残差がすべて 0 の場合。
    """
    y_res, t_res = residuals.y_res, residuals.t_res
    n = len(y_res)
    if n < 3:
        raise ValueError(f"行数が少なすぎます: {n}")
    sum_tt = float(np.sum(t_res**2))
    if sum_tt == 0:
        raise ValueError("処置の残差がすべて 0 です（処置が共変量で完全に説明されている）")
    estimate = float(np.sum(t_res * y_res) / sum_tt)
    error = y_res - estimate * t_res
    dof = n - 1
    sigma2 = float(np.sum(error**2) / dof)
    return ResidualRegression(
        estimate=estimate,
        std_error=float(np.sqrt(sigma2 / sum_tt)),
        dof=dof,
        residual_sd_y=float(np.sqrt(sigma2)),
        residual_sd_t=float(np.sqrt(sum_tt / dof)),
    )


def partial_r2_from_t(t_value: ArrayLike, dof: int) -> ArrayLike:
    """t 値から部分決定係数 t² / (t² + dof) を求める。"""
    return t_value**2 / (t_value**2 + dof)


def robustness_value(t_value: float, dof: int, q: float = 1.0, alpha: float | None = None) -> float:
    """robustness value を求める（sensemakr の `robustness_value` と同じ式）。

    Args:
        t_value: t 値。
        dof: 自由度。
        q: 推定値を減らす割合（1 なら 0 にする、0.5 なら半分にする）。
        alpha: None なら点推定の RV_q、値を渡すと CI（有意水準 alpha）の RV_q,α。

    Returns:
        r2dz = r2yz としたとき、推定値（または CI）が (1 − q) 倍の値に届く交絡の強さ。

    Raises:
        ValueError: `q` が正でない、または `alpha` が (0, 1) にない場合。
    """
    if q <= 0:
        raise ValueError(f"q は正の値にしてください: {q}")
    fq = q * abs(t_value) / np.sqrt(dof)
    if alpha is None:
        return float(0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2))
    if not 0 < alpha < 1:
        raise ValueError(f"alpha は 0 と 1 の間にしてください: {alpha}")
    f_crit = abs(stats.t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
    fqa = fq - f_crit
    if fqa <= 0:
        # もともと CI が (1 − q) 倍の値を含んでいる
        return 0.0
    if fq > 1 / f_crit:
        # 極端に t 値が大きいときの式（sensemakr の場合分け）
        return float((fq**2 - f_crit**2) / (1 + fq**2))
    return float(0.5 * (np.sqrt(fqa**4 + 4 * fqa**2) - fqa**2))


def _check_r2(r2dz: ArrayLike, r2yz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """部分決定係数が r2dz ∈ [0, 1)・r2yz ∈ [0, 1] にあるか確かめて、配列で返す。"""
    dz = np.asarray(r2dz, dtype=float)
    yz = np.asarray(r2yz, dtype=float)
    if np.any((dz < 0) | (dz >= 1)) or np.any((yz < 0) | (yz > 1)):
        raise ValueError("部分決定係数は r2dz を [0, 1)、r2yz を [0, 1] にしてください")
    return dz, yz


def bias(fit: ResidualRegression, r2dz: ArrayLike, r2yz: ArrayLike) -> np.ndarray:
    """未観測交絡（r2dz, r2yz）による推定値の偏りの大きさ（絶対値）。"""
    dz, yz = _check_r2(r2dz, r2yz)
    return fit.std_error * np.sqrt(fit.dof) * np.sqrt(yz * dz / (1 - dz))


def adjusted_estimate(
    fit: ResidualRegression, r2dz: ArrayLike, r2yz: ArrayLike, reduce: bool = True
) -> np.ndarray:
    """未観測交絡を調整した推定値。

    Args:
        fit: 残差回帰の結果。
        r2dz: R²_{T~Z|X}。
        r2yz: R²_{Y~Z|T,X}。
        reduce: True なら推定値を 0 に近づける向き（保守的）、False なら 0 から遠ざける向きに
            ずらす。

    Returns:
        調整後の推定値。
    """
    sign = np.sign(fit.estimate) if fit.estimate != 0 else 1.0
    direction = -1.0 if reduce else 1.0
    return fit.estimate + direction * sign * bias(fit, r2dz, r2yz)


def adjusted_std_error(fit: ResidualRegression, r2dz: ArrayLike, r2yz: ArrayLike) -> np.ndarray:
    """未観測交絡を調整したときの標準誤差（Z を加えて自由度が1減ることも考える）。"""
    dz, yz = _check_r2(r2dz, r2yz)
    return fit.std_error * np.sqrt((1 - yz) / (1 - dz) * fit.dof / (fit.dof - 1))


def adjusted_interval(
    fit: ResidualRegression,
    r2dz: ArrayLike,
    r2yz: ArrayLike,
    alpha: float = 0.05,
    reduce: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """未観測交絡を調整したときの CI（下限, 上限）。"""
    estimate = adjusted_estimate(fit, r2dz, r2yz, reduce)
    se = adjusted_std_error(fit, r2dz, r2yz)
    critical = stats.t.ppf(1 - alpha / 2, fit.dof - 1)
    return estimate - critical * se, estimate + critical * se


def adjusted_t_value(
    fit: ResidualRegression,
    r2dz: ArrayLike,
    r2yz: ArrayLike,
    reduce: bool = True,
    null_hypothesis: float = 0.0,
) -> np.ndarray:
    """未観測交絡を調整したときの t 値（帰無仮説 `null_hypothesis` に対する）。

    r2yz = 1（結果の残差をすべて説明する交絡）では標準誤差が 0 になり、t 値は ±inf（または NaN）。
    """
    estimate = adjusted_estimate(fit, r2dz, r2yz, reduce)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (estimate - null_hypothesis) / adjusted_std_error(fit, r2dz, r2yz)


def sensitivity_statistics(
    fit: ResidualRegression, q: float = 1.0, alpha: float = 0.05
) -> dict[str, float]:
    """感度分析の要約（推定値・CI・R²_{Y~T|X}・RV_q・RV_q,α）を返す。

    Args:
        fit: 残差回帰の結果。
        q: 推定値を減らす割合（RV の計算に使う）。
        alpha: CI の有意水準。

    Returns:
        要約の dict（表の1行にする）。
    """
    critical = stats.t.ppf(1 - alpha / 2, fit.dof)
    return {
        "estimate": fit.estimate,
        "std_error": fit.std_error,
        "ci_lower": fit.estimate - critical * fit.std_error,
        "ci_upper": fit.estimate + critical * fit.std_error,
        "t_value": fit.t_value,
        "dof": fit.dof,
        "partial_r2_yd": fit.partial_r2_yd,
        "q": q,
        "rv_q": robustness_value(fit.t_value, fit.dof, q=q),
        "rv_qa": robustness_value(fit.t_value, fit.dof, q=q, alpha=alpha),
        "alpha": alpha,
    }


def benchmark_bounds(
    r2dxj: float, r2yxj: float, kd: ArrayLike = 1.0, ky: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """観測された共変量 X_j の kd・ky 倍の強さの交絡に当たる (r2dz, r2yz) を求める。

    sensemakr の `ovb_partial_r2_bound` と同じ式。kd は処置への、ky は結果への強さの倍率で、
    「Z は X_j と比べて、処置の残差を kd 倍、結果の残差を ky 倍説明する」と考える。

    Args:
        r2dxj: R²_{T~X_j|X_−j}（他の共変量を除いた後に X_j が処置を説明する割合）。
        r2yxj: R²_{Y~X_j|T,X_−j}（同じく結果を説明する割合）。
        kd: 処置への強さの倍率。
        ky: 結果への強さの倍率（None なら kd と同じ）。

    Returns:
        (r2dz, r2yz)。その強さの交絡があり得ない（r2dz ≥ 1 など）ときは NaN。r2yz は 1 で頭打ち。
    """
    kd_arr = np.asarray(kd, dtype=float)
    ky_arr = kd_arr if ky is None else np.asarray(ky, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2dz = kd_arr * r2dxj / (1 - r2dxj)
        r2zxj = kd_arr * r2dxj**2 / ((1 - kd_arr * r2dxj) * (1 - r2dxj))
        r2yz = ((np.sqrt(ky_arr) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (
            r2yxj / (1 - r2yxj)
        )
    invalid = (r2dz >= 1) | (r2zxj >= 1) | (r2zxj < 0) | ~np.isfinite(r2yz)
    r2dz = np.where(invalid, np.nan, r2dz)
    r2yz = np.where(invalid, np.nan, np.minimum(r2yz, 1.0))
    return r2dz, r2yz


def _variance_explained(full: np.ndarray, reduced: np.ndarray) -> float:
    """1 − Var(full) / Var(reduced)（0 未満は 0 にする。推定のぶれで負になりうるため）。"""
    return float(max(0.0, 1.0 - np.mean(full**2) / np.mean(reduced**2)))


def benchmark_partial_r2(
    y: np.ndarray | pd.Series,
    t: np.ndarray | pd.Series,
    controls: pd.DataFrame,
    benchmarks: Mapping[str, Sequence[str]],
    *,
    model_y: BaseEstimator,
    model_t: BaseEstimator,
    cv: Splitter,
    full_residuals: DmlResiduals,
) -> pd.DataFrame:
    """観測された共変量（またはそのグループ）ごとに、処置・結果への部分決定係数を求める。

    共変量を除いて nuisance を cross-fitting し直し、残差の分散がどれだけ増えるかで測る。

    - R²_{T~X_j|X_−j} = 1 − Var(T̃) / Var(T̃_−j)
    - R²_{Y~X_j|T,X_−j} = 1 − Var(e) / Var(e_−j)（e は残差回帰の残差 Ỹ − θ̂·T̃）

    Args:
        y: 結果変数。
        t: 処置変数。
        controls: 観測された共変量（`[X, W]`。列名で benchmark を指定する）。
        benchmarks: benchmark の名前 → 除く列のリスト（例: {"X1": ["X1"], "X1+X2": ["X1", "X2"]}）。
        model_y: E[Y|X,W] のモデル（`full_residuals` を作ったのと同じもの）。
        model_t: E[T|X,W] のモデル。
        cv: 分割器（`full_residuals` を作ったのと同じもの）。
        full_residuals: すべての共変量を使った残差（`residualize` の結果）。

    Returns:
        benchmark ごとの `r2dxj`・`r2yxj` の表。

    Raises:
        KeyError: 共変量に無い列を指定した場合。
        ValueError: benchmark がすべての列を除く場合。
    """
    full_fit = fit_residual_regression(full_residuals)
    full_error = full_residuals.y_res - full_fit.estimate * full_residuals.t_res
    rows = []
    for name, columns in benchmarks.items():
        absent = [c for c in columns if c not in controls.columns]
        if absent:
            raise KeyError(f"共変量に無い列です: {absent}")
        reduced = controls.drop(columns=list(columns))
        if reduced.shape[1] == 0:
            raise ValueError(f"benchmark {name} はすべての共変量を除いてしまいます")
        residuals = residualize(y, t, reduced, model_y=model_y, model_t=model_t, cv=cv)
        reduced_fit = fit_residual_regression(residuals)
        reduced_error = residuals.y_res - reduced_fit.estimate * residuals.t_res
        rows.append(
            {
                "benchmark": name,
                "columns": ",".join(columns),
                "r2dxj": _variance_explained(full_residuals.t_res, residuals.t_res),
                "r2yxj": _variance_explained(full_error, reduced_error),
            }
        )
    return pd.DataFrame(rows)


def benchmark_table(
    fit: ResidualRegression,
    benchmarks: pd.DataFrame,
    multipliers: Sequence[float] = (1.0, 2.0, 3.0),
    alpha: float = 0.05,
    reduce: bool = True,
) -> pd.DataFrame:
    """benchmark の k 倍の強さの交絡があるときの、調整後の推定値・CI・t 値の表。

    Args:
        fit: 残差回帰の結果。
        benchmarks: `benchmark_partial_r2` の結果（`benchmark`・`r2dxj`・`r2yxj` の列）。
        multipliers: 強さの倍率 k（kd = ky = k）。
        alpha: CI の有意水準。
        reduce: 推定値を 0 に近づける向きに調整するか。

    Returns:
        benchmark・倍率ごとの表（`label` は図の注記用。例: "1×X3"）。
    """
    rows = []
    items = zip(
        benchmarks["benchmark"].astype(str),
        benchmarks["r2dxj"].astype(float),
        benchmarks["r2yxj"].astype(float),
        strict=True,
    )
    for name, r2dxj, r2yxj in items:
        for k in multipliers:
            r2dz, r2yz = (float(v) for v in benchmark_bounds(r2dxj, r2yxj, kd=k))
            row: dict[str, float | str] = {
                "benchmark": name,
                "label": f"{k:g}×{name}",
                "k": k,
                "r2dz": r2dz,
                "r2yz": r2yz,
            }
            if np.isfinite(r2dz) and np.isfinite(r2yz):
                lower, upper = adjusted_interval(fit, r2dz, r2yz, alpha, reduce)
                row |= {
                    "adjusted_estimate": float(adjusted_estimate(fit, r2dz, r2yz, reduce)),
                    "adjusted_se": float(adjusted_std_error(fit, r2dz, r2yz)),
                    "adjusted_t": float(adjusted_t_value(fit, r2dz, r2yz, reduce)),
                    "adjusted_lower": float(lower),
                    "adjusted_upper": float(upper),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def partial_r2_grid(
    fit: ResidualRegression,
    r2_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    reduce: bool = True,
) -> pd.DataFrame:
    """(r2dz, r2yz) の格子ごとの調整後の推定値・CI・t 値（等高線図用、縦長の表）。

    Args:
        fit: 残差回帰の結果。
        r2_values: 格子の値（r2dz・r2yz に共通。r2dz は 1 未満）。
        alpha: CI の有意水準。
        reduce: 推定値を 0 に近づける向きに調整するか。

    Returns:
        `r2dz`・`r2yz`・`estimate`・`lower`・`upper`・`t_value` の表。
    """
    values = np.asarray(r2_values, dtype=float)
    dz, yz = (a.ravel() for a in np.meshgrid(values, values))
    lower, upper = adjusted_interval(fit, dz, yz, alpha, reduce)
    return pd.DataFrame(
        {
            "r2dz": dz,
            "r2yz": yz,
            "estimate": adjusted_estimate(fit, dz, yz, reduce),
            "lower": lower,
            "upper": upper,
            "t_value": adjusted_t_value(fit, dz, yz, reduce),
        }
    )


def linear_partial_r2(target: np.ndarray | pd.Series, features: pd.DataFrame) -> pd.Series:
    """線形回帰 target ~ features で、各列の部分決定係数（t² / (t² + dof)）を求める。

    benchmark にする共変量の候補を、処置との関連の強い順に選ぶための簡便な目安。

    Args:
        target: 目的変数（処置など）。
        features: 説明変数。

    Returns:
        列ごとの部分決定係数（大きい順）。
    """
    check_finite("目的変数", target)
    check_finite("説明変数", features)
    result = sm.OLS(
        np.asarray(target, dtype=float), sm.add_constant(features, has_constant="add")
    ).fit()
    tvalues = result.tvalues.drop("const")
    r2 = partial_r2_from_t(tvalues, int(result.df_resid))
    return pd.Series(r2, index=features.columns, name="partial_r2").sort_values(ascending=False)
