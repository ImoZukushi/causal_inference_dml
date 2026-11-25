"""econml の DML 推定器の生成・学習と、ATE・CATE の取り出し。

推定器は部分線形モデル Y = θ(X)·T + g(X, W) + ε, T = m(X, W) + η を前提とする DML。

- `linear_dml`: `LinearDML`。θ(X) を X の線形関数で表す。係数と CI を解析的に出せる。
- `causal_forest_dml`: `CausalForestDML`。θ(X) を因果フォレストで非線形に推定する。CI あり。

どちらも nuisance モデルを cross-fitting で学習し、out-of-fold の残差だけを最終段に使う。
分割は `cv_strategy` で選ぶ: "kfold"（行をシャッフルした K-fold。`mc_iters` 回繰り返せる）か
"timeseries"（時間順の連続ブロック。検証ブロックの前後 `gap` 行を学習から除く）。
X は効果修飾かつ交絡因子、W は nuisance にだけ使う共変量。
T は連続値（`discrete_treatment=False`）で、効果は「T を1単位増やしたときの Y の変化」。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.base import BaseEstimator

from causal_inference.nuisance import BlockedTimeSeriesSplit, check_finite, make_cv

EstimatorFactory = Callable[..., Any]
# econml の cv 引数: 分割数（econml が K-fold を作る）か分割器
CrossFitting = int | BlockedTimeSeriesSplit


def _build_linear_dml(
    model_y: BaseEstimator, model_t: BaseEstimator, cv: CrossFitting, mc_iters: int, seed: int
) -> LinearDML:
    return LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,
        cv=cv,
        mc_iters=mc_iters,
        random_state=seed,
    )


def _build_causal_forest_dml(
    model_y: BaseEstimator,
    model_t: BaseEstimator,
    cv: CrossFitting,
    mc_iters: int,
    seed: int,
    n_estimators: int = 500,
    min_samples_leaf: int = 10,
    max_depth: int | None = None,
) -> CausalForestDML:
    # CI を出すには木の数が subforest_size（既定4）の倍数である必要がある
    n_estimators = max(4, n_estimators - n_estimators % 4)
    return CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,
        cv=cv,
        mc_iters=mc_iters,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=seed,
    )


# 推定器の名前 → 生成関数（引数: model_y, model_t, cv, mc_iters, seed, **追加の引数）
ESTIMATORS: dict[str, EstimatorFactory] = {
    "linear_dml": _build_linear_dml,
    "causal_forest_dml": _build_causal_forest_dml,
}
ESTIMATOR_LABELS = {"linear_dml": "LinearDML", "causal_forest_dml": "CausalForestDML"}


def build_estimator(
    name: str,
    model_y: BaseEstimator,
    model_t: BaseEstimator,
    *,
    n_splits: int = 5,
    mc_iters: int = 1,
    seed: int = 0,
    cv_strategy: str = "kfold",
    gap: int = 0,
    **kwargs: Any,
) -> Any:
    """名前を指定して DML 推定器を作る（cross-fitting の設定込み）。

    - `kfold`: econml には分割数（int）を渡す。econml は `random_state` からシャッフルした K-fold を
      作り、`mc_iters` 回繰り返すときは繰り返しごとに別の分割にする。
    - `timeseries`: `BlockedTimeSeriesSplit` を渡す。行が時間順に並んでいる必要がある。分割は乱数に
      よらず決まるので、`mc_iters` は1だけ（繰り返しても同じ分割になり意味が無い）。

    Args:
        name: 推定器の名前（`ESTIMATORS` のキー）。
        model_y: 結果の nuisance モデル E[Y|X,W]（未学習）。
        model_t: 処置の nuisance モデル E[T|X,W]（未学習）。
        n_splits: cross-fitting の分割数（2以上）。
        mc_iters: cross-fitting を分割を変えて繰り返す回数（1以上）。nuisance の予測は平均する。
        seed: 乱数シード。
        cv_strategy: 分割の方法（"kfold" / "timeseries"）。
        gap: `timeseries` で検証ブロックの前後から学習に使わない行数。
        **kwargs: 推定器ごとの追加の引数（例: CausalForestDML の `n_estimators`）。

    Returns:
        未学習の econml 推定器。

    Raises:
        KeyError: 未登録の名前の場合。
        ValueError: `n_splits < 2`（cross-fitting にならない）、`mc_iters < 1`、未知の
            `cv_strategy`、または `timeseries` で `mc_iters > 1` の場合。
    """
    if name not in ESTIMATORS:
        raise KeyError(f"未登録の推定器です: {name}（登録済み: {sorted(ESTIMATORS)}）")
    # 分割数・分割方法の検査（n_splits < 2 などなら ValueError）
    splitter = make_cv(n_splits, seed, strategy=cv_strategy, gap=gap)
    if mc_iters < 1:
        raise ValueError(f"mc_iters は1以上にしてください: {mc_iters}")
    if cv_strategy == "timeseries" and mc_iters > 1:
        raise ValueError("timeseries の分割は毎回同じなので、mc_iters は1にしてください")
    cv: CrossFitting = splitter if isinstance(splitter, BlockedTimeSeriesSplit) else n_splits
    return ESTIMATORS[name](model_y, model_t, cv, mc_iters, seed, **kwargs)


def _as_2d(values: pd.DataFrame | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def fit_estimator(
    estimator: Any,
    y: pd.Series | np.ndarray,
    t: pd.Series | np.ndarray,
    x: pd.DataFrame | np.ndarray,
    w: pd.DataFrame | np.ndarray | None = None,
    *,
    cache_values: bool = False,
) -> Any:
    """推定器を学習する（入力に欠損・inf が無いことを確かめてから）。

    Args:
        estimator: `build_estimator` で作った推定器。
        y: 結果変数。
        t: 処置変数。
        x: 効果修飾かつ交絡因子（X）。
        w: nuisance にだけ使う共変量（W）。
        cache_values: nuisance の残差を保持するか（`refit_final` や一部の感度分析で必要）。

    Returns:
        学習済みの推定器（`estimator` と同じオブジェクト）。

    Raises:
        ValueError: 入力に欠損・inf がある場合、または行数が揃っていない場合。
    """
    arrays = {"Y": y, "T": t, "X": x, "W": w}
    for name, values in arrays.items():
        check_finite(name, None if values is None else np.asarray(values, dtype=float))
    n_rows = {name: len(values) for name, values in arrays.items() if values is not None}
    if len(set(n_rows.values())) != 1:
        raise ValueError(f"Y・T・X・W の行数が揃っていません: {n_rows}")
    estimator.fit(
        np.asarray(y, dtype=float).ravel(),
        np.asarray(t, dtype=float).ravel(),
        X=_as_2d(x),
        W=_as_2d(w),
        cache_values=cache_values,
    )
    return estimator


def tune_causal_forest(
    estimator: CausalForestDML,
    y: pd.Series | np.ndarray,
    t: pd.Series | np.ndarray,
    x: pd.DataFrame | np.ndarray,
    w: pd.DataFrame | np.ndarray | None = None,
    params: dict[str, list[Any]] | str = "auto",
) -> dict[str, Any]:
    """CausalForestDML の最終段（因果フォレスト）のパラメータを R-score で選ぶ（econml の `tune`）。

    学習データの 70% で小さいフォレストを学習し、残りで R-score を比べてグリッドから選ぶ。
    選んだ値は `estimator` に設定される（学習はされないので、後で `fit_estimator` を呼ぶ）。

    Args:
        estimator: 未学習の `CausalForestDML`。
        y: 結果変数。
        t: 処置変数。
        x: X。
        w: W。
        params: パラメータのグリッド（{名前: [候補, ...]}）。"auto" なら econml の既定のグリッド。

    Returns:
        選ばれたパラメータ（グリッドに含まれる名前だけ）。

    Raises:
        TypeError: `estimator` が `CausalForestDML` でない場合。
        ValueError: 入力に欠損・inf がある場合。
    """
    if not isinstance(estimator, CausalForestDML):
        raise TypeError("tune_causal_forest は CausalForestDML 専用です")
    for name, values in {"Y": y, "T": t, "X": x, "W": w}.items():
        check_finite(name, None if values is None else np.asarray(values, dtype=float))
    estimator.tune(
        np.asarray(y, dtype=float).ravel(),
        np.asarray(t, dtype=float).ravel(),
        X=_as_2d(x),
        W=_as_2d(w),
        params=params,
    )
    names = (
        ["min_weight_fraction_leaf", "max_depth", "min_var_fraction_leaf"]
        if params == "auto"
        else list(params)
    )
    return {name: getattr(estimator, name) for name in names}


@dataclass
class EffectSummary:
    """1つの推定器の推定結果。

    Attributes:
        name: 推定器の名前。
        ate: ATE の点推定（CATE の標本平均）。
        ate_lower: ATE の CI の下限。
        ate_upper: ATE の CI の上限。
        cate: 行ごとの CATE（列 `cate`, `cate_lower`, `cate_upper`）。
        alpha: CI の有意水準（`1 - alpha` の CI）。
        base_treatment: 処置の基準値 T0。
        target_treatment: 処置の目標値 T1（効果は T を T0 から T1 に変えたときの Y の変化）。
    """

    name: str
    ate: float
    ate_lower: float
    ate_upper: float
    cate: pd.DataFrame
    alpha: float
    base_treatment: float = 0.0
    target_treatment: float = 1.0

    def to_row(self) -> dict[str, Any]:
        """ATE の表の1行（推定器の比較用）を返す。"""
        return {
            "estimator": self.name,
            "base_treatment": self.base_treatment,
            "target_treatment": self.target_treatment,
            "ate": self.ate,
            "ate_lower": self.ate_lower,
            "ate_upper": self.ate_upper,
            "ci_level": 1 - self.alpha,
            "cate_mean": float(self.cate["cate"].mean()),
            "cate_std": float(self.cate["cate"].std()),
        }

    def rescaled(self, factor: float) -> EffectSummary:
        """効果（ATE・CI・CATE の全列）を `factor` 倍した新しい結果を返す。

        Y を定数 c 倍して推定したとき、`factor=1/c` で元の単位に戻す（`scaling` を参照）。

        Args:
            factor: 倍率（正の値。負だと CI の下限・上限が入れ替わるので受け付けない）。

        Returns:
            倍率を掛けた推定結果。

        Raises:
            ValueError: `factor` が正の有限な値でない場合。
        """
        if not (np.isfinite(factor) and factor > 0):
            raise ValueError(f"factor は正の有限な値にしてください: {factor}")
        return replace(
            self,
            ate=self.ate * factor,
            ate_lower=self.ate_lower * factor,
            ate_upper=self.ate_upper * factor,
            cate=self.cate * factor,
        )


def ci_level_label(alpha: float) -> str:
    """有意水準 `alpha` の CI の信頼水準の表示名（例: 0.05 → "95"、0.1 → "90"、0.005 → "99.5"）。"""
    return f"{(1 - alpha) * 100:g}"


def ci_columns(alpha: float) -> tuple[str, str]:
    """有意水準 `alpha` の CATE の CI の列名（下限, 上限）。

    例: 0.05 → ("cate_lower_95", "cate_upper_95")。
    """
    label = ci_level_label(alpha)
    return f"cate_lower_{label}", f"cate_upper_{label}"


def summarize_effects(
    name: str,
    estimator: Any,
    x: pd.DataFrame | np.ndarray,
    alpha: float = 0.05,
    extra_alphas: Sequence[float] = (),
    t0: float = 0.0,
    t1: float = 1.0,
) -> EffectSummary:
    """学習済みの推定器から ATE と行ごとの CATE（点推定と CI）を取り出す。

    効果は処置を基準値 `t0` から目標値 `t1` に変えたときの Y の変化
    （econml の `effect(X, T0, T1)`）。
    部分線形モデル（DML）では効果は T について線形なので、θ(X)·(t1 − t0) になり、
    差 t1 − t0 だけで決まる（基準値の水準によらない）。

    CATE の表の列:
        - `cate`: 点推定
        - `cate_lower` / `cate_upper`: 有意水準 `alpha` の CI（ATE の CI と同じ水準）
        - `cate_lower_<水準>` / `cate_upper_<水準>`: `alpha` と `extra_alphas` のそれぞれの CI
          （例: alpha=0.05 → `cate_lower_95`、0.01 → `cate_lower_99`。`ci_columns` で列名を作れる）

    Args:
        name: 推定器の名前（表示用）。
        estimator: 学習済みの推定器。
        x: CATE を求める行の X（ATE はこの行の平均）。
        alpha: CI の有意水準（ATE と `cate_lower`/`cate_upper` に使う）。
        extra_alphas: CATE の CI を追加で求める有意水準（例: (0.01,) で 99% CI）。
        t0: 処置の基準値（base treatment）。
        t1: 処置の目標値（target treatment）。

    Returns:
        推定結果。

    Raises:
        ValueError: `x` に欠損・inf がある場合、有意水準が 0 より大きく 1 未満でない場合、
            または `t0`・`t1` が使えない値の場合（`check_treatment_change`）。
    """
    alphas = list(dict.fromkeys([alpha, *extra_alphas]))  # 順序を保って重複を除く
    for value in alphas:
        if not 0 < value < 1:
            raise ValueError(f"有意水準は 0 より大きく 1 未満にしてください: {value}")
    t0, t1 = check_treatment_change(t0, t1)
    x_array = _as_2d(x)
    check_finite("X", x_array)
    change = {"T0": t0, "T1": t1}
    cate = np.asarray(estimator.effect(x_array, **change)).ravel()
    columns: dict[str, np.ndarray] = {"cate": cate}
    for value in alphas:
        lower, upper = (
            np.asarray(v).ravel() for v in estimator.effect_interval(x_array, **change, alpha=value)
        )
        lower_col, upper_col = ci_columns(value)
        columns[lower_col], columns[upper_col] = lower, upper
        if value == alpha:
            columns["cate_lower"], columns["cate_upper"] = lower, upper
    ate_lower, ate_upper = (
        float(np.ravel(v)[0]) for v in estimator.ate_interval(x_array, **change, alpha=alpha)
    )
    index = x.index if isinstance(x, pd.DataFrame) else None
    # 基本の列（cate, cate_lower, cate_upper）を先頭にし、水準ごとの CI を続ける
    order = ["cate", "cate_lower", "cate_upper", *(c for a in alphas for c in ci_columns(a))]
    return EffectSummary(
        name=name,
        ate=float(np.ravel(estimator.ate(x_array, **change))[0]),
        ate_lower=ate_lower,
        ate_upper=ate_upper,
        cate=pd.DataFrame(columns, index=index)[order],
        alpha=alpha,
        base_treatment=t0,
        target_treatment=t1,
    )


def check_treatment_change(t0: float, t1: float) -> tuple[float, float]:
    """処置の基準値 T0 と目標値 T1 が使える値か確かめて、float で返す。

    Args:
        t0: 処置の基準値。
        t1: 処置の目標値。

    Returns:
        (t0, t1)。

    Raises:
        ValueError: 有限な値でない、または同じ値（効果が常に 0 になる）の場合。
    """
    t0, t1 = float(t0), float(t1)
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise ValueError(f"処置の基準値・目標値は有限な値にしてください: {t0}, {t1}")
    if t0 == t1:
        raise ValueError(f"処置の基準値と目標値が同じです（効果は常に 0 になる）: {t0}")
    return t0, t1


def treatment_change_label(t0: float, t1: float, treatment_name: str = "T") -> str:
    """処置の変化の表示名（例: "T: 0 → 1"）。図のタイトル・軸に使う。"""
    return f"{treatment_name}: {t0:g} → {t1:g}"


def nuisance_scores(estimator: Any) -> pd.DataFrame:
    """学習済みの推定器の、fold ごとの nuisance モデルの決定係数（out-of-fold の R²）を返す。

    Args:
        estimator: 学習済みの推定器。

    Returns:
        列 `model`（"model_y" / "model_t"）, `mc_iter`, `fold`, `r2` の DataFrame。
    """
    rows = []
    for model, scores in (
        ("model_y", estimator.nuisance_scores_y),
        ("model_t", estimator.nuisance_scores_t),
    ):
        for mc_iter, fold_scores in enumerate(scores):
            for fold, score in enumerate(np.ravel(fold_scores)):
                rows.append({"model": model, "mc_iter": mc_iter, "fold": fold, "r2": float(score)})
    return pd.DataFrame(rows)


def linear_dml_coefficients(
    estimator: LinearDML, feature_names: list[str], alpha: float = 0.05
) -> pd.DataFrame:
    """LinearDML の θ(X) = θ0 + Σ βj·Xj の係数と CI を返す。

    Args:
        estimator: 学習済みの `LinearDML`。
        feature_names: X の列名。
        alpha: CI の有意水準。

    Returns:
        列 `term`, `coef`, `lower`, `upper` の DataFrame（先頭行が切片 θ0）。
    """
    coef = np.ravel(estimator.coef_)
    lower, upper = (np.ravel(v) for v in estimator.coef__interval(alpha=alpha))
    intercept = float(np.ravel(estimator.intercept_)[0])
    i_lower, i_upper = (float(np.ravel(v)[0]) for v in estimator.intercept__interval(alpha=alpha))
    return pd.DataFrame(
        {
            "term": ["intercept", *feature_names],
            "coef": [intercept, *coef],
            "lower": [i_lower, *lower],
            "upper": [i_upper, *upper],
        }
    )
