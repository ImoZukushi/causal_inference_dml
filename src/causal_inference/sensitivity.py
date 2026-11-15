"""推定した効果の感度分析（未観測交絡に対する頑健さ）と反証テスト。

1. E-value（VanderWeele & Ding, 2017）: 推定した効果を「無かったこと」にするために、
   未観測の交絡因子が処置と結果の両方に持つ必要がある関連の強さ（リスク比のスケール）の最小値。
   連続の結果変数には、標準化した効果 d = 効果 / SD(Y) からリスク比を RR ≈ exp(0.91·d) で近似する
   （VanderWeele, 2017 の連続アウトカム向けの近似）。T も連続なので、T を `delta` 動かしたときの効果
   （PLR では θ·delta）を使う。DoWhy の E-value は線形回帰・GLM の推定器専用で econml の推定器には
   使えないため、ここで計算する。
2. omitted variable bias の感度分析（Chernozhukov et al., 2022。econml の実装）:
   未観測の交絡因子が、結果の残差の分散を c_y、処置の残差（Riesz representer）の分散を
   c_t の割合だけ説明するとき、ATE がどこまで動きうるかの範囲と、効果が有意でなくなる
   交絡の強さ（robustness value）。
3. DoWhy の反証テスト（refuter）: 偽の処置・偽の結果・ランダムな共変量・部分標本・
   模擬的な未観測交絡で推定し直し、推定値が期待どおりに振る舞うかを見る。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from causal_inference.nuisance import check_finite

# 連続の結果変数の標準化効果 d をリスク比に換算する係数（RR ≈ exp(0.91·d)）
SMD_TO_LOG_RR = 0.91


# ---------------------------------------------------------------------------
# E-value
# ---------------------------------------------------------------------------
def e_value_from_rr(rr: float) -> float:
    """リスク比の E-value を返す（RR < 1 は 1/RR にしてから計算する）。

    Args:
        rr: リスク比（正の値）。

    Returns:
        E-value（1以上）。`rr` が非有限・0以下なら NaN。
    """
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    rr = 1.0 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value_continuous(
    estimate: float,
    lower: float,
    upper: float,
    sd_outcome: float,
    delta: float = 1.0,
) -> dict[str, float]:
    """連続の結果変数・連続の処置について、効果の点推定と CI の E-value を返す。

    Args:
        estimate: T を1単位動かしたときの効果（ATE）の点推定。
        lower: その CI の下限。
        upper: その CI の上限。
        sd_outcome: 結果変数の標準偏差（効果の標準化に使う）。
        delta: E-value を計算する処置の変化幅（例: T の標準偏差。効果は `estimate · delta`）。

    Returns:
        `delta`, `effect`（estimate·delta）, `smd`（標準化効果）, `rr`（換算したリスク比）,
        `e_value_point`（点推定の E-value）, `e_value_ci`（CI の 0 に近い側の限界の E-value。
        CI が 0 を含めば 1）の dict。入力が非有限なら値は NaN。

    Raises:
        ValueError: `sd_outcome` または `delta` が正でない場合。
    """
    if not (sd_outcome > 0 and delta > 0):
        raise ValueError(f"sd_outcome と delta は正の値にしてください: {sd_outcome}, {delta}")
    values = np.array([estimate, lower, upper], dtype=float) * delta / sd_outcome
    if not np.all(np.isfinite(values)):
        nan = float("nan")
        return {"delta": delta, "effect": estimate * delta, "smd": nan, "rr": nan,
                "e_value_point": nan, "e_value_ci": nan}  # fmt: skip
    smd, smd_lower, smd_upper = values
    if smd_lower <= 0 <= smd_upper:
        # CI が効果なしを含む: 交絡が無くても有意でないので、CI の E-value は 1
        e_ci = 1.0
    else:
        # CI の、効果なし（0）に近い側の限界
        limit = smd_lower if smd > 0 else smd_upper
        e_ci = e_value_from_rr(float(np.exp(SMD_TO_LOG_RR * limit)))
    rr = float(np.exp(SMD_TO_LOG_RR * smd))
    return {
        "delta": delta,
        "effect": estimate * delta,
        "smd": float(smd),
        "rr": rr,
        "e_value_point": e_value_from_rr(rr),
        "e_value_ci": e_ci,
    }


def e_value_table(
    ate_table: pd.DataFrame, sd_outcome: float, deltas: dict[str, float]
) -> pd.DataFrame:
    """推定器ごと・処置の変化幅ごとの E-value の表を返す。

    Args:
        ate_table: 列 `estimator`, `ate`, `ate_lower`, `ate_upper` を持つ表。
        sd_outcome: 結果変数の標準偏差。
        deltas: 処置の変化幅（{表示名: 幅}。例 {"T +1単位": 1.0, "T +1SD": sd_t}）。

    Returns:
        列 `estimator`, `delta_label` と `e_value_continuous` の各値の DataFrame。
    """
    rows = []
    for _, row in ate_table.iterrows():
        estimate, lower, upper = (float(row[c]) for c in ("ate", "ate_lower", "ate_upper"))
        for label, delta in deltas.items():
            result = e_value_continuous(estimate, lower, upper, sd_outcome, delta)
            rows.append({"estimator": row["estimator"], "delta_label": label, **result})
    return pd.DataFrame(rows)


def e_value_bias_curve(rr: float, n_points: int = 200, max_rr: float | None = None) -> pd.DataFrame:
    """推定したリスク比 `rr` をちょうど打ち消す、交絡因子の2つの強さの組の曲線を返す。

    処置と交絡因子のリスク比 RR_TU、交絡因子と結果のリスク比 RR_UY がこの曲線より上（右上）に
    あれば、効果を打ち消しうる。曲線は RR_TU·RR_UY / (RR_TU + RR_UY − 1) = rr を満たし、
    RR_TU = RR_UY のときが E-value。

    Args:
        rr: リスク比（1未満なら 1/rr を使う）。
        n_points: 曲線の点の数。
        max_rr: 横軸の上限（None なら E-value の3倍）。

    Returns:
        列 `rr_treatment_confounder`, `rr_confounder_outcome` の DataFrame（rr が 1 なら空）。
    """
    rr = 1.0 / rr if rr < 1 else rr
    if not np.isfinite(rr) or rr <= 1:
        return pd.DataFrame(columns=["rr_treatment_confounder", "rr_confounder_outcome"])
    upper = max_rr if max_rr is not None else 3 * e_value_from_rr(rr)
    a = np.linspace(rr * 1.0001, upper, n_points)
    b = rr * (a - 1) / (a - rr)
    keep = b <= upper
    return pd.DataFrame({"rr_treatment_confounder": a[keep], "rr_confounder_outcome": b[keep]})


# ---------------------------------------------------------------------------
# omitted variable bias の感度分析（econml）
# ---------------------------------------------------------------------------
def ovb_summary(
    estimator: Any, c_y: float = 0.05, c_t: float = 0.05, rho: float = 1.0, alpha: float = 0.05
) -> dict[str, float]:
    """1組の交絡の強さ（c_y, c_t）での ATE の範囲と、robustness value を返す（econml）。

    Args:
        estimator: 学習済みの `LinearDML` または `CausalForestDML`。
        c_y: 未観測交絡が結果の残差の分散を説明する割合（0以上1未満）。
        c_t: 未観測交絡が処置の残差（Riesz representer）の分散を説明する割合（0以上1未満）。
        rho: 2つの交絡の効果の相関（1 が最悪のケース）。
        alpha: CI の有意水準。

    Returns:
        `theta`（PLR の ATE）, `theta_lower` / `theta_upper`（交絡を考えた点推定の範囲）,
        `ci_lower` / `ci_upper`（さらに標本誤差も考えた範囲）, `rv_theta` / `rv_ci`
        （点推定 / CI が 0 を含むようになる交絡の強さ c_y = c_t）の dict。
    """
    theta_lower, theta_upper = estimator.sensitivity_interval(
        alpha=alpha, c_y=c_y, c_t=c_t, rho=rho, interval_type="theta"
    )
    ci_lower, ci_upper = estimator.sensitivity_interval(
        alpha=alpha, c_y=c_y, c_t=c_t, rho=rho, interval_type="ci"
    )
    params = estimator._ortho_learner_model_final._model_final.sensitivity_params
    return {
        "c_y": c_y,
        "c_t": c_t,
        "rho": rho,
        "theta": float(params.theta),
        "theta_lower": float(theta_lower),
        "theta_upper": float(theta_upper),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "rv_theta": float(estimator.robustness_value(alpha=alpha, interval_type="theta")),
        "rv_ci": float(estimator.robustness_value(alpha=alpha, interval_type="ci")),
    }


def ovb_grid(
    estimator: Any,
    c_values: Sequence[float] = tuple(np.round(np.arange(0.0, 0.31, 0.02), 2)),
    rho: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """c_y × c_t の格子の各点での ATE の範囲を返す（等高線図用）。

    Args:
        estimator: 学習済みの `LinearDML` または `CausalForestDML`。
        c_values: c_y・c_t の候補（0以上1未満）。
        rho: 2つの交絡の効果の相関。
        alpha: CI の有意水準。

    Returns:
        列 `c_y`, `c_t`, `theta_lower`, `theta_upper`, `ci_lower`, `ci_upper` の DataFrame。
    """
    rows = []
    for c_y in c_values:
        for c_t in c_values:
            theta = estimator.sensitivity_interval(
                alpha=alpha, c_y=c_y, c_t=c_t, rho=rho, interval_type="theta"
            )
            ci = estimator.sensitivity_interval(
                alpha=alpha, c_y=c_y, c_t=c_t, rho=rho, interval_type="ci"
            )
            rows.append(
                {
                    "c_y": float(c_y),
                    "c_t": float(c_t),
                    "theta_lower": float(theta[0]),
                    "theta_upper": float(theta[1]),
                    "ci_lower": float(ci[0]),
                    "ci_upper": float(ci[1]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DoWhy の反証テスト
# ---------------------------------------------------------------------------
# 反証テストの種類ごとの「期待される結果」
#   zero: 新しい推定値が 0 に近いはず（偽の処置・偽の結果）
#   same: 新しい推定値が元の推定値に近いはず（無関係な共変量の追加・部分標本）
#   shift: 期待値は無い（模擬的な未観測交絡でどれだけ動くかを見る）
EXPECTATIONS = {
    "placebo_treatment_refuter": "zero",
    "dummy_outcome_refuter": "zero",
    "random_common_cause": "same",
    "data_subset_refuter": "same",
    "add_unobserved_common_cause": "shift",
}


@dataclass(frozen=True)
class RefuterSpec:
    """1つの反証テストの設定。

    Attributes:
        label: 表示名。
        method_name: DoWhy の refuter の名前（`EXPECTATIONS` のキー）。
        kwargs: `refute_estimate` に渡す引数。
    """

    label: str
    method_name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def default_refuters(
    num_simulations: int = 10,
    seed: int = 0,
    n_jobs: int = 1,
    subset_fraction: float = 0.8,
    unobserved_strengths: Sequence[float] = (0.05, 0.1, 0.2),
) -> list[RefuterSpec]:
    """標準の反証テストの組を返す。

    Args:
        num_simulations: 偽の処置・偽の結果・ランダムな共変量・部分標本で推定し直す回数。
        seed: 乱数シード。
        n_jobs: 推定し直しの並列数（DoWhy の joblib）。
        subset_fraction: 部分標本の割合。
        unobserved_strengths: 模擬的な未観測交絡が処置・結果に効く強さ（線形の係数。DoWhy の
            direct-simulation）の候補。1つにつき1回推定し直す。

    Returns:
        反証テストの設定のリスト。
    """
    common = {"num_simulations": num_simulations, "random_state": seed, "n_jobs": n_jobs}
    specs = [
        RefuterSpec(
            "偽の処置（T を並べ替え）",
            "placebo_treatment_refuter",
            {**common, "placebo_type": "permute"},
        ),
        RefuterSpec("無関係な共変量を追加", "random_common_cause", dict(common)),
        RefuterSpec(
            f"部分標本（{subset_fraction:.0%}）",
            "data_subset_refuter",
            {**common, "subset_fraction": subset_fraction},
        ),
        # dummy_outcome_refuter は random_state を受け取らない（NumPy の大域の乱数を使う）
        RefuterSpec(
            "偽の結果（Y を無関係なノイズに）",
            "dummy_outcome_refuter",
            {"num_simulations": num_simulations},
        ),
    ]
    for strength in unobserved_strengths:
        specs.append(
            RefuterSpec(
                f"模擬的な未観測交絡（強さ {strength}）",
                "add_unobserved_common_cause",
                {
                    "simulation_method": "direct-simulation",
                    "confounders_effect_on_treatment": "linear",
                    "confounders_effect_on_outcome": "linear",
                    "effect_strength_on_treatment": strength,
                    "effect_strength_on_outcome": strength,
                },
            )
        )
    return specs


def fit_dowhy_wrapper(
    estimator: Any,
    y: pd.Series | np.ndarray,
    t: pd.Series | np.ndarray,
    x: pd.DataFrame,
    w: pd.DataFrame | None = None,
    *,
    outcome_name: str = "Y",
    treatment_name: str = "T",
) -> Any:
    """econml の推定器を DoWhy の因果モデルとして学習し直す（反証テストの準備）。

    DoWhy は X と W の両方を処置と結果の共通原因（backdoor 調整集合）とし、X を効果修飾とする。
    推定器は `estimator` の設定（nuisance モデル・cross-fitting）で、DoWhy の中で改めて学習される。

    Args:
        estimator: econml の推定器（学習済みでなくてよい。設定だけを使う）。
        y: 結果変数。
        t: 処置変数。
        x: X（列名を DoWhy の変数名に使う）。
        w: W。
        outcome_name: 結果変数の名前。
        treatment_name: 処置変数の名前。

    Returns:
        学習済みの `DoWhyWrapper`（`refute_estimate` を呼べる）。

    Raises:
        ValueError: 入力に欠損・inf がある場合。
    """
    for name, values in {"Y": y, "T": t, "X": x, "W": w}.items():
        check_finite(name, None if values is None else np.asarray(values, dtype=float))
    # `estimator.dowhy` は参照のたびに新しいラッパーを作るので、1つを変数に持って使う
    wrapper = estimator.dowhy
    wrapper.fit(
        np.asarray(y, dtype=float).ravel(),
        np.asarray(t, dtype=float).ravel(),
        X=x.to_numpy(dtype=float),
        W=None if w is None else w.to_numpy(dtype=float),
        outcome_names=[outcome_name],
        treatment_names=[treatment_name],
        feature_names=list(x.columns),
        confounder_names=[] if w is None else list(w.columns),
    )
    return wrapper


def _judge(expectation: str, original: float, new: float, tolerance: float) -> str:
    """反証テストの結果の目安（"OK" / "要確認" / "-"）を返す。"""
    if not (np.isfinite(original) and np.isfinite(new)):
        return "-"
    scale = max(abs(original), 1e-12)
    if expectation == "zero":
        return "OK" if abs(new) <= tolerance * scale else "要確認"
    if expectation == "same":
        return "OK" if abs(new - original) <= tolerance * scale else "要確認"
    return "-"


def run_refuters(
    wrapper: Any,
    refuters: Sequence[RefuterSpec],
    *,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    """反証テストを順に実行し、結果の表を返す（失敗したテストはエラー文を残して続ける）。

    Args:
        wrapper: `fit_dowhy_wrapper` の結果。
        refuters: 反証テストの設定。
        tolerance: 目安の判定の許容幅（元の推定値の絶対値に対する割合）。
            期待が「0 に近い」なら |新しい推定値| ≤ tolerance·|元の推定値|、
            「元と同じ」なら |差| ≤ tolerance·|元の推定値|。

    Returns:
        列 `refuter`, `method_name`, `expectation`, `estimated_effect`, `new_effect`, `p_value`,
        `judgement`, `elapsed_sec`, `error` の DataFrame。

    Notes:
        `estimated_effect` は元の推定値（偽の結果のテストでは、偽の結果に対する期待値の 0）。
        `p_value` は DoWhy の有意性検定（シミュレーションの分布に対する元の推定値の位置）で、
        シミュレーション回数が少ないと不安定なので、`judgement` と合わせて見る。
    """
    original_effect = float(np.ravel(wrapper.estimate_.value)[0])
    rows = []
    for spec in refuters:
        expectation = EXPECTATIONS.get(spec.method_name, "shift")
        start = time.perf_counter()
        row: dict[str, Any] = {
            "refuter": spec.label,
            "method_name": spec.method_name,
            "expectation": expectation,
        }
        try:
            result = wrapper.refute_estimate(method_name=spec.method_name, **spec.kwargs)
            # 偽の結果のテストはリストで返る
            result = result[0] if isinstance(result, list) else result
            new_effect = float(np.ravel(result.new_effect)[0])
            p_value = (result.refutation_result or {}).get("p_value", np.nan)
            row.update(
                estimated_effect=float(np.ravel(result.estimated_effect)[0]),
                new_effect=new_effect,
                p_value=float(np.ravel(p_value)[0]) if p_value is not None else np.nan,
                judgement=_judge(expectation, original_effect, new_effect, tolerance),
                error="",
            )
        except Exception as error:  # noqa: BLE001  1つのテストの失敗で全体を止めない
            row.update(
                estimated_effect=original_effect,
                new_effect=np.nan,
                p_value=np.nan,
                judgement="-",
                error=f"{type(error).__name__}: {error}",
            )
        row["elapsed_sec"] = time.perf_counter() - start
        rows.append(row)
    columns = ["refuter", "method_name", "expectation", "estimated_effect", "new_effect",
               "p_value", "judgement", "elapsed_sec", "error"]  # fmt: skip
    return pd.DataFrame(rows, columns=columns)
