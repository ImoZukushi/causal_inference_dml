"""推定結果の評価（ベースラインの推定と、真の CATE との比較）。

真の CATE との比較は、真値が分かる合成データでだけ使える。実データでは使えない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_inference.estimators import EffectSummary, check_treatment_change
from causal_inference.nuisance import check_finite


def baseline_estimates(
    y: pd.Series | np.ndarray,
    t: pd.Series | np.ndarray,
    x: pd.DataFrame | np.ndarray,
    w: pd.DataFrame | np.ndarray | None = None,
    alpha: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
) -> pd.DataFrame:
    """DML と比べるためのベースライン（OLS による T の係数から求めた効果）を返す。

    - `naive_ols`: Y ~ T（調整なし。交絡があるとずれる）
    - `adjusted_ols`: Y ~ T + X + W（線形の調整。効果の異質性・非線形の交絡は扱えない）

    効果は処置を `t0` から `t1` に変えたときの Y の変化（T の係数 × (t1 − t0)）。

    Args:
        y: 結果変数。
        t: 処置変数。
        x: X。
        w: W。
        alpha: CI の有意水準。
        t0: 処置の基準値。
        t1: 処置の目標値。

    Returns:
        列 `estimator`, `base_treatment`, `target_treatment`, `ate`, `ate_lower`, `ate_upper`,
        `ci_level` の DataFrame。

    Raises:
        ValueError: 入力に欠損・inf がある場合、または `t0`・`t1` が使えない値の場合。
    """
    t0, t1 = check_treatment_change(t0, t1)
    y_array = np.asarray(y, dtype=float).ravel()
    t_array = np.asarray(t, dtype=float).ravel()
    covariates = [np.asarray(x, dtype=float)]
    if w is not None:
        covariates.append(np.asarray(w, dtype=float))
    for name, values in {"Y": y_array, "T": t_array, "X・W": np.hstack(covariates)}.items():
        check_finite(name, values)

    rows = []
    for name, design in (
        ("naive_ols", t_array.reshape(-1, 1)),
        ("adjusted_ols", np.column_stack([t_array, *covariates])),
    ):
        # HC1（不均一分散に頑健）の標準誤差を使う
        fitted = sm.OLS(y_array, sm.add_constant(design)).fit(cov_type="HC1")
        # T の係数（1単位あたり）を T0→T1 の効果にする（t1 < t0 なら CI の上下が入れ替わる）
        delta = t1 - t0
        bounds = sorted(float(b) * delta for b in fitted.conf_int(alpha=alpha)[1])
        rows.append(
            {
                "estimator": name,
                "base_treatment": t0,
                "target_treatment": t1,
                "ate": float(fitted.params[1]) * delta,
                "ate_lower": bounds[0],
                "ate_upper": bounds[1],
                "ci_level": 1 - alpha,
            }
        )
    return pd.DataFrame(rows)


def compare_with_truth(
    summary: EffectSummary, true_cate: pd.Series | np.ndarray
) -> dict[str, float | str]:
    """推定した CATE を真の CATE と比べる（合成データ専用）。

    Args:
        summary: 推定結果。
        true_cate: 行ごとの真の CATE（`summary.cate` と同じ行の並び）。

    Returns:
        指標の dict:
        `ate_true`（真の CATE の標本平均）, `ate_error`（推定 − 真）,
        `ate_covered`（ATE の CI が真値を含むか）, `cate_rmse`, `cate_corr`（ピアソン相関）,
        `cate_coverage`（CATE の CI が真値を含む行の割合）, `cate_interval_width`（CI の幅の平均）。

    Raises:
        ValueError: 行数が合わない場合。
    """
    truth = np.asarray(true_cate, dtype=float).ravel()
    cate = summary.cate
    if len(truth) != len(cate):
        raise ValueError(f"真の CATE の行数 {len(truth)} が推定結果の行数 {len(cate)} と違います")
    estimate = cate["cate"].to_numpy()
    # 真値が欠損・inf の行は比べない
    valid = np.isfinite(truth) & np.isfinite(estimate)
    truth, estimate = truth[valid], estimate[valid]
    lower, upper = cate["cate_lower"].to_numpy()[valid], cate["cate_upper"].to_numpy()[valid]
    ate_true = float(truth.mean())
    return {
        "estimator": summary.name,
        "ate_true": ate_true,
        "ate_estimate": summary.ate,
        "ate_error": summary.ate - ate_true,
        "ate_covered": bool(summary.ate_lower <= ate_true <= summary.ate_upper),
        "cate_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "cate_corr": float(np.corrcoef(estimate, truth)[0, 1]) if truth.std() > 0 else np.nan,
        "cate_coverage": float(np.mean((lower <= truth) & (truth <= upper))),
        "cate_interval_width": float(np.mean(upper - lower)),
    }
