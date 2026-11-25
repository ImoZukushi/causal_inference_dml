"""src/causal_inference/partial_r2.py（部分決定係数による感度分析）のテスト。

式の値は R の sensemakr の Darfur の例（Cinelli & Hazlett, 2020）と照合する。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from causal_inference import plots  # noqa: E402
from causal_inference.nuisance import BlockedTimeSeriesSplit, cross_fit_predict  # noqa: E402
from causal_inference.partial_r2 import (  # noqa: E402
    ResidualRegression,
    adjusted_estimate,
    adjusted_interval,
    adjusted_std_error,
    benchmark_bounds,
    benchmark_partial_r2,
    benchmark_table,
    fit_residual_regression,
    linear_partial_r2,
    partial_r2_grid,
    residualize,
    robustness_value,
    sensitivity_statistics,
)

# sensemakr の Darfur の例（directlyharmed の係数）
DARFUR = ResidualRegression(
    estimate=0.09731582, std_error=0.02325654, dof=783, residual_sd_y=1.0, residual_sd_t=1.0
)


def test_darfur_statistics() -> None:
    """t 値・R²_{Y~D|X}・RV が sensemakr の値と一致する。"""
    stats = sensitivity_statistics(DARFUR, q=1.0, alpha=0.05)
    assert stats["t_value"] == pytest.approx(4.18445, rel=1e-4)
    assert stats["partial_r2_yd"] == pytest.approx(0.02187309, rel=1e-4)
    assert stats["rv_q"] == pytest.approx(0.1387764, rel=1e-4)
    assert stats["rv_qa"] == pytest.approx(0.07625797, rel=1e-4)


def test_darfur_adjusted_estimate() -> None:
    """1×female の強さの交絡で調整した推定値が sensemakr の値と一致する。"""
    r2dz, r2yz = 0.00916428, 0.12464092
    assert float(adjusted_estimate(DARFUR, r2dz, r2yz)) == pytest.approx(0.07522362, abs=1e-5)
    assert float(adjusted_std_error(DARFUR, r2dz, r2yz)) == pytest.approx(0.02187952, rel=1e-3)
    lower, upper = adjusted_interval(DARFUR, r2dz, r2yz, alpha=0.05)
    assert float(lower) == pytest.approx(0.03227214, abs=2e-4)
    assert float(upper) == pytest.approx(0.1181751, abs=2e-4)


@pytest.mark.parametrize("q", [1.0, 0.5])
def test_robustness_value_is_the_boundary(q: float) -> None:
    """r2dz = r2yz = RV_q で推定値が (1 − q) 倍、RV_q,α で CI の限界が (1 − q) 倍になる。"""
    rv = robustness_value(DARFUR.t_value, DARFUR.dof, q=q)
    assert float(adjusted_estimate(DARFUR, rv, rv)) == pytest.approx(
        (1 - q) * DARFUR.estimate, abs=1e-8
    )
    rv_a = robustness_value(DARFUR.t_value, DARFUR.dof, q=q, alpha=0.05)
    lower, _ = adjusted_interval(DARFUR, rv_a, rv_a, alpha=0.05)
    assert float(lower) == pytest.approx((1 - q) * DARFUR.estimate, abs=1e-6)


def test_robustness_value_zero_when_not_significant() -> None:
    """もともと CI が 0 を含めば RV_q,α は 0。"""
    assert robustness_value(1.0, 500, alpha=0.05) == 0.0
    with pytest.raises(ValueError, match="q は正"):
        robustness_value(3.0, 500, q=0)


def test_adjusted_estimate_moves_toward_zero_for_negative_effect() -> None:
    """負の効果では、調整後の推定値は 0 に近づく（大きくなる）。"""
    negative = DARFUR.rescaled(-1.0)
    adjusted = float(adjusted_estimate(negative, 0.05, 0.05))
    assert negative.estimate < adjusted < 0
    assert float(adjusted_estimate(negative, 0.05, 0.05, reduce=False)) < negative.estimate


def test_invalid_partial_r2_raises() -> None:
    """r2dz = 1 などはエラーにする。"""
    with pytest.raises(ValueError, match="部分決定係数"):
        adjusted_estimate(DARFUR, 1.0, 0.1)


def test_rescaled_keeps_unit_free_statistics() -> None:
    """倍率を掛けても t 値の大きさ・R²・RV は変わらず、推定値・標準誤差は倍率倍になる。"""
    scaled = DARFUR.rescaled(-1000.0)
    assert scaled.estimate == pytest.approx(-1000 * DARFUR.estimate)
    assert scaled.std_error == pytest.approx(1000 * DARFUR.std_error)
    assert scaled.partial_r2_yd == pytest.approx(DARFUR.partial_r2_yd)
    base, new = sensitivity_statistics(DARFUR), sensitivity_statistics(scaled)
    assert new["rv_qa"] == pytest.approx(base["rv_qa"])
    assert new["ci_lower"] == pytest.approx(-1000 * base["ci_upper"])
    with pytest.raises(ValueError, match="factor"):
        DARFUR.rescaled(0.0)


def test_benchmark_bounds() -> None:
    """r2dz は kd について線形で、あり得ない強さは NaN にする。"""
    r2dz, r2yz = benchmark_bounds(0.1, 0.05, kd=[1.0, 2.0])
    assert r2dz[1] == pytest.approx(2 * r2dz[0])
    assert r2yz[1] > r2yz[0]
    impossible_dz, impossible_yz = benchmark_bounds(0.6, 0.05, kd=2.0)
    assert np.isnan(impossible_dz) and np.isnan(impossible_yz)


def test_partial_r2_grid() -> None:
    """格子の原点は交絡なしの推定値で、交絡が強いほど推定値は 0 に近づく。"""
    grid = partial_r2_grid(DARFUR, np.linspace(0, 0.4, 5))
    assert len(grid) == 25
    origin = grid[(grid["r2dz"] == 0) & (grid["r2yz"] == 0)].iloc[0]
    assert origin["estimate"] == pytest.approx(DARFUR.estimate)
    corner = grid[(grid["r2dz"] == 0.4) & (grid["r2yz"] == 0.4)].iloc[0]
    assert corner["estimate"] < 0 < origin["lower"]


def _confounded_data(n: int = 2000, seed: int = 0) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """X1 が強い交絡因子、X2 が弱い交絡因子、W1 が結果だけに効く線形のデータ（効果は 2）。"""
    rng = np.random.default_rng(seed)
    controls = pd.DataFrame(rng.normal(size=(n, 3)), columns=["X1", "X2", "W1"])
    t = 1.0 * controls["X1"] + 0.2 * controls["X2"] + rng.normal(size=n)
    y = 2.0 * t + 1.0 * controls["X1"] + 0.2 * controls["X2"] + controls["W1"] + rng.normal(size=n)
    return y, t, controls


def test_residualize_and_benchmarks() -> None:
    """残差回帰で効果を推定でき、強い交絡因子ほど benchmark の部分決定係数が大きい。"""
    y, t, controls = _confounded_data()
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = LinearRegression()
    residuals = residualize(y, t, controls, model_y=model, model_t=model, cv=cv)
    fit = fit_residual_regression(residuals)
    assert fit.estimate == pytest.approx(2.0, abs=0.1)
    assert fit.dof == len(y) - 1

    benchmarks = benchmark_partial_r2(
        y,
        t,
        controls,
        {"X1": ["X1"], "X2": ["X2"], "W1": ["W1"]},
        model_y=model,
        model_t=model,
        cv=cv,
        full_residuals=residuals,
    )
    r2 = benchmarks.set_index("benchmark")
    # T = X1 + 0.2·X2 + η なので R²_{T~X1|X2,W1} ≈ 1/2、R²_{T~X2|X1,W1} ≈ 0.04/1.04
    assert r2.loc["X1", "r2dxj"] == pytest.approx(0.5, abs=0.05)
    assert r2.loc["X2", "r2dxj"] == pytest.approx(0.04 / 1.04, abs=0.02)
    assert r2.loc["W1", "r2dxj"] < 0.01
    assert r2.loc["X1", "r2yxj"] > r2.loc["X2", "r2yxj"]

    table = benchmark_table(fit, benchmarks, multipliers=(1.0, 2.0))
    assert len(table) == 6
    assert {"label", "r2dz", "r2yz", "adjusted_estimate", "adjusted_lower"} <= set(table.columns)
    assert table.loc[table["label"] == "1×X2", "adjusted_estimate"].iloc[0] < fit.estimate

    with pytest.raises(KeyError, match="共変量に無い列"):
        benchmark_partial_r2(
            y, t, controls, {"Z": ["Z"]}, model_y=model, model_t=model, cv=cv,
            full_residuals=residuals,
        )  # fmt: skip


def test_residualize_rejects_nonfinite() -> None:
    """欠損を含む入力はエラーにする。"""
    y, t, controls = _confounded_data(n=100)
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="欠損または inf"):
        residualize(y, t, controls, model_y=LinearRegression(), model_t=LinearRegression(),
                    cv=KFold(2))  # fmt: skip


def test_cross_fit_predict_with_timeseries_split() -> None:
    """時系列の分割でも全行の out-of-fold の予測ができる。"""
    y, _, controls = _confounded_data(n=300)
    prediction = cross_fit_predict(
        LinearRegression(), controls.to_numpy(), y.to_numpy(), BlockedTimeSeriesSplit(3, gap=5)
    )
    assert np.isfinite(prediction).all()


def test_cross_fit_predict_requires_full_coverage() -> None:
    """検証側に入らない行がある分割器はエラーにする。"""

    class FirstHalf:
        def split(self, features: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
            n = len(features)
            return [(np.arange(n // 2, n), np.arange(n // 2))]

    with pytest.raises(ValueError, match="一度も入らない行"):
        cross_fit_predict(LinearRegression(), np.ones((10, 1)), np.ones(10), FirstHalf())  # type: ignore[arg-type]


def test_linear_partial_r2_ranks_strong_confounder_first() -> None:
    """処置との線形の部分決定係数は、強い交絡因子が最も大きい。"""
    _, t, controls = _confounded_data()
    ranking = linear_partial_r2(t, controls)
    assert ranking.index[0] == "X1"
    assert ranking["X1"] == pytest.approx(0.5, abs=0.05)


def test_plot_partial_r2_contours() -> None:
    """等高線図は2枚の軸（と色の凡例）を持ち、benchmark の点を描ける。"""
    grid = partial_r2_grid(DARFUR, np.linspace(0, 0.4, 11))
    benchmarks = pd.DataFrame(
        {"label": ["1×female", "3×female"], "r2dz": [0.009, 0.027], "r2yz": [0.125, 0.374]}
    )
    fig = plots.plot_partial_r2_contours(
        grid, DARFUR.estimate, benchmarks=benchmarks, treatment_change="T: 0 → 1"
    )
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 2
    assert "下限" in titled[1].get_title()
    plt.close(fig)
