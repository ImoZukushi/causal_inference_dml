"""src/causal_inference（DML による CATE の推定と感度分析）のテスト。

学習を伴うテストは、小さいデータ・小さいモデルで高速に動かす（推定の精度ではなく、形と振る舞いを見る）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import generate_dummy_dml as gen
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from causal_inference import plots  # noqa: E402
from causal_inference.estimators import (  # noqa: E402
    build_estimator,
    ci_columns,
    ci_level_label,
    fit_estimator,
    linear_dml_coefficients,
    nuisance_scores,
    summarize_effects,
    tune_causal_forest,
)
from causal_inference.evaluation import baseline_estimates, compare_with_truth  # noqa: E402
from causal_inference.nuisance import (  # noqa: E402
    DEFAULT_NUISANCE_PARAMS,
    BlockedTimeSeriesSplit,
    check_finite,
    make_cv,
    make_nuisance_model,
    tune_nuisance,
)
from causal_inference.output import OutputSaver  # noqa: E402
from causal_inference.sensitivity import (  # noqa: E402
    RefuterSpec,
    default_refuters,
    e_value_bias_curve,
    e_value_continuous,
    e_value_from_rr,
    e_value_table,
    fit_dowhy_wrapper,
    ovb_grid,
    ovb_summary,
    run_refuters,
)

SMALL_MODEL = {"n_estimators": 60, "num_leaves": 15, "learning_rate": 0.1, "n_jobs": 2}


@pytest.fixture(scope="module")
def dataset() -> SimpleNamespace:
    result = gen.make_dummy_dml(n=2000, seed=0)
    data = result.data
    return SimpleNamespace(
        y=data["Y"],
        t=data["T"],
        x=data[gen.CONFOUNDER_COLS],
        w=data[gen.COVARIATE_COLS],
        true_cate=result.truth["true_cate"].to_numpy(),
    )


def _build(name: str, **kwargs) -> object:
    return build_estimator(
        name,
        make_nuisance_model(SMALL_MODEL),
        make_nuisance_model(SMALL_MODEL),
        n_splits=3,
        seed=0,
        **kwargs,
    )


@pytest.fixture(scope="module")
def linear_dml(dataset: SimpleNamespace) -> object:
    return fit_estimator(_build("linear_dml"), dataset.y, dataset.t, dataset.x, dataset.w)


@pytest.fixture(scope="module")
def causal_forest(dataset: SimpleNamespace) -> object:
    estimator = _build("causal_forest_dml", n_estimators=40)
    return fit_estimator(estimator, dataset.y, dataset.t, dataset.x, dataset.w)


# ---------------------------------------------------------------------------
# cross-fitting・入力の検査
# ---------------------------------------------------------------------------
def test_cross_fitting_requires_at_least_two_splits() -> None:
    assert make_cv(5, 0).get_n_splits() == 5
    with pytest.raises(ValueError, match="n_splits"):
        make_cv(1, 0)
    with pytest.raises(ValueError, match="n_splits"):
        build_estimator("linear_dml", make_nuisance_model(), make_nuisance_model(), n_splits=1)
    with pytest.raises(ValueError, match="mc_iters"):
        build_estimator("linear_dml", make_nuisance_model(), make_nuisance_model(), mc_iters=0)
    with pytest.raises(KeyError):
        build_estimator("unknown", make_nuisance_model(), make_nuisance_model())


def test_estimators_use_requested_cross_fitting() -> None:
    estimator = build_estimator(
        "causal_forest_dml", make_nuisance_model(), make_nuisance_model(), n_splits=4, mc_iters=2
    )
    assert estimator.cv == 4 and estimator.mc_iters == 2
    # CI を出すため、木の数は 4 の倍数に丸める
    assert (
        build_estimator(
            "causal_forest_dml", make_nuisance_model(), make_nuisance_model(), n_estimators=50
        ).n_estimators
        == 48
    )


@pytest.mark.parametrize("gap", [0, 7])
def test_blocked_time_series_split(gap: int) -> None:
    n_rows, n_splits = 103, 4
    folds = list(BlockedTimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.zeros(n_rows)))
    assert len(folds) == n_splits
    tests = [test for _, test in folds]
    # 検証は時間順の連続ブロックで、全行がちょうど1回ずつ検証に入る
    np.testing.assert_array_equal(np.concatenate(tests), np.arange(n_rows))
    for train, test in folds:
        assert np.all(np.diff(test) == 1)
        assert not set(train) & set(test)
        # 検証ブロックの前後 gap 行は学習に使わない
        start, stop = test[0], test[-1] + 1
        near = set(range(max(0, start - gap), start)) | set(range(stop, min(n_rows, stop + gap)))
        assert not set(train) & near
        # それ以外の行（過去と未来）はすべて学習に使う
        assert len(train) == n_rows - len(test) - len(near)


def test_blocked_time_series_split_validation() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        BlockedTimeSeriesSplit(n_splits=1)
    with pytest.raises(ValueError, match="gap"):
        BlockedTimeSeriesSplit(n_splits=3, gap=-1)
    with pytest.raises(ValueError, match="行数"):
        list(BlockedTimeSeriesSplit(n_splits=5).split(np.zeros(3)))
    with pytest.raises(ValueError, match="gap"):
        list(BlockedTimeSeriesSplit(n_splits=2, gap=100).split(np.zeros(10)))


def test_make_cv_strategies() -> None:
    assert isinstance(make_cv(5, 0), KFold)
    splitter = make_cv(5, 0, strategy="timeseries", gap=3)
    assert isinstance(splitter, BlockedTimeSeriesSplit) and splitter.gap == 3
    with pytest.raises(ValueError, match="未知"):
        make_cv(5, 0, strategy="random")
    with pytest.raises(ValueError, match="gap"):
        make_cv(5, 0, strategy="kfold", gap=2)


def test_time_series_cross_fitting_in_estimators(dataset: SimpleNamespace) -> None:
    estimator = _build("linear_dml", cv_strategy="timeseries", gap=10)
    assert isinstance(estimator.cv, BlockedTimeSeriesSplit) and estimator.cv.gap == 10
    with pytest.raises(ValueError, match="mc_iters"):
        _build("linear_dml", cv_strategy="timeseries", mc_iters=2)

    fit_estimator(estimator, dataset.y, dataset.t, dataset.x, dataset.w)
    summary = summarize_effects("LinearDML", estimator, dataset.x)
    assert summary.ate == pytest.approx(1.0, abs=0.25)
    assert len(nuisance_scores(estimator)) == 2 * 3

    # Optuna も同じ時系列の分割で探索できる
    result = tune_nuisance(
        pd.concat([dataset.x, dataset.w], axis=1).iloc[:600], dataset.y.iloc[:600],
        cv=make_cv(3, strategy="timeseries", gap=5), n_trials=2,
        base_params={"n_estimators": 30, "n_jobs": 2},
    )  # fmt: skip
    assert len(result.trials) == 2

    # DoWhy の反証テストも、時系列の分割を持つ推定器で動く
    rows = slice(0, 400)
    wrapper = fit_dowhy_wrapper(
        _build("linear_dml", cv_strategy="timeseries", gap=5),
        dataset.y.iloc[rows], dataset.t.iloc[rows], dataset.x.iloc[rows], dataset.w.iloc[rows],
    )  # fmt: skip
    table = run_refuters(wrapper, default_refuters(num_simulations=2)[:1])
    assert (table["error"] == "").all(), table["error"].tolist()

    fig = plots.plot_cv_folds(make_cv(4, strategy="timeseries", gap=20), 500)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_rejected(dataset: SimpleNamespace, bad: float) -> None:
    with pytest.raises(ValueError, match="欠損または inf"):
        check_finite("X", np.array([1.0, bad]))
    x = dataset.x.copy()
    x.iloc[0, 0] = bad
    with pytest.raises(ValueError, match="X に欠損または inf"):
        fit_estimator(_build("linear_dml"), dataset.y, dataset.t, x, dataset.w)


def test_row_counts_must_match(dataset: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="行数"):
        fit_estimator(_build("linear_dml"), dataset.y, dataset.t.iloc[:-1], dataset.x, dataset.w)


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
def test_tune_nuisance_keeps_base_and_fixed_params(dataset: SimpleNamespace) -> None:
    features = pd.concat([dataset.x, dataset.w], axis=1).iloc[:600]
    target = dataset.y.iloc[:600]
    # 既定は学習率が極端に小さく、ほとんど学習しない（探索した方が必ず良くなる）
    base = {"n_estimators": 30, "n_jobs": 2, "learning_rate": 1e-4}

    def space(trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "max_depth": 3,
        }

    result = tune_nuisance(
        features, target, cv=make_cv(3, 0), n_trials=3, target_name="Y",
        search_space=space, base_params=base,
    )  # fmt: skip
    assert len(result.trials) == 3 and (result.trials["target"] == "Y").all()
    assert not result.used_baseline and result.best_value < result.baseline_value
    # 固定のパラメータ・探索空間の固定値・探索した値がすべて入る
    assert result.best_params["n_jobs"] == 2 and result.best_params["n_estimators"] == 30
    assert result.best_params["max_depth"] == 3 and 4 <= result.best_params["num_leaves"] <= 16
    assert set(DEFAULT_NUISANCE_PARAMS) <= set(result.best_params)
    assert result.best_value == pytest.approx(result.trials["value"].min())
    fig = plots.plot_tuning_history([result])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    with pytest.raises(ValueError, match="n_trials"):
        tune_nuisance(features, target, cv=make_cv(3, 0), n_trials=0)


def test_tune_nuisance_falls_back_to_base_params(dataset: SimpleNamespace) -> None:
    features = pd.concat([dataset.x, dataset.w], axis=1).iloc[:600]
    base = {"n_estimators": 30, "n_jobs": 2, "learning_rate": 0.1}

    def bad_space(trial) -> dict:
        # どの試行もほとんど学習しない（既定より悪い）
        return {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4)}

    result = tune_nuisance(
        features, dataset.y.iloc[:600], cv=make_cv(3, 0), n_trials=2,
        search_space=bad_space, base_params=base,
    )  # fmt: skip
    assert result.used_baseline
    assert result.best_params == {**DEFAULT_NUISANCE_PARAMS, **base}
    assert result.best_value == result.baseline_value


# ---------------------------------------------------------------------------
# 推定
# ---------------------------------------------------------------------------
def test_linear_dml_recovers_ate_and_reports_cate(
    dataset: SimpleNamespace, linear_dml: object
) -> None:
    summary = summarize_effects("LinearDML", linear_dml, dataset.x)
    assert summary.ate == pytest.approx(1.0, abs=0.25)
    assert summary.ate_lower < summary.ate < summary.ate_upper
    assert list(summary.cate.columns) == [
        "cate", "cate_lower", "cate_upper", "cate_lower_95", "cate_upper_95",
    ]  # fmt: skip
    assert len(summary.cate) == len(dataset.x)
    assert (summary.cate["cate_lower"] <= summary.cate["cate_upper"]).all()
    # X1 は CATE を大きくし、X2 は小さくする（真の θ の係数は +0.5・-0.4）
    coefficients = linear_dml_coefficients(linear_dml, gen.CONFOUNDER_COLS).set_index("term")
    assert len(coefficients) == 16
    assert coefficients.loc["X1", "coef"] > 0.2 and coefficients.loc["X2", "coef"] < -0.1
    scores = nuisance_scores(linear_dml)
    assert len(scores) == 2 * 3 and set(scores["model"]) == {"model_y", "model_t"}


def test_cate_intervals_at_multiple_levels(dataset: SimpleNamespace, causal_forest: object) -> None:
    assert ci_level_label(0.05) == "95" and ci_level_label(0.01) == "99"
    assert ci_level_label(0.005) == "99.5"
    assert ci_columns(0.01) == ("cate_lower_99", "cate_upper_99")

    summary = summarize_effects("CF", causal_forest, dataset.x, alpha=0.05, extra_alphas=[0.01])
    cate = summary.cate
    assert list(cate.columns) == [
        "cate", "cate_lower", "cate_upper",
        "cate_lower_95", "cate_upper_95", "cate_lower_99", "cate_upper_99",
    ]  # fmt: skip
    # 基本の CI は alpha の CI と同じ
    np.testing.assert_array_equal(cate["cate_lower"], cate["cate_lower_95"])
    # 99% CI は 95% CI を含み、どちらも点推定を含む
    assert (cate["cate_lower_99"] <= cate["cate_lower_95"]).all()
    assert (cate["cate_upper_95"] <= cate["cate_upper_99"]).all()
    assert (cate["cate_lower_95"] <= cate["cate"]).all() and (
        cate["cate"] <= cate["cate_upper_95"]
    ).all()
    # 同じ水準を重ねて渡しても列は重複しない
    again = summarize_effects("CF", causal_forest, dataset.x, extra_alphas=[0.05, 0.01, 0.01])
    assert list(again.cate.columns) == list(cate.columns)
    with pytest.raises(ValueError, match="有意水準"):
        summarize_effects("CF", causal_forest, dataset.x, extra_alphas=[1.0])


def test_causal_forest_estimates_and_tunes(dataset: SimpleNamespace, causal_forest: object) -> None:
    summary = summarize_effects("CausalForestDML", causal_forest, dataset.x)
    metrics = compare_with_truth(summary, dataset.true_cate)
    assert metrics["cate_corr"] > 0.3
    assert len(causal_forest.feature_importances_) == len(gen.CONFOUNDER_COLS)

    estimator = _build("causal_forest_dml", n_estimators=40)
    chosen = tune_causal_forest(
        estimator,
        dataset.y.iloc[:800],
        dataset.t.iloc[:800],
        dataset.x.iloc[:800],
        dataset.w.iloc[:800],
        params={"max_depth": [3, None]},
    )
    assert set(chosen) == {"max_depth"} and estimator.max_depth == chosen["max_depth"]
    with pytest.raises(TypeError):
        tune_causal_forest(_build("linear_dml"), dataset.y, dataset.t, dataset.x)


def test_baseline_naive_is_biased_and_compare_with_truth(dataset: SimpleNamespace) -> None:
    table = baseline_estimates(dataset.y, dataset.t, dataset.x, dataset.w).set_index("estimator")
    # 合成データは交絡で調整なしの推定が上にずれる
    assert table.loc["naive_ols", "ate"] > table.loc["adjusted_ols", "ate"] + 0.2

    perfect = SimpleNamespace(
        name="perfect",
        ate=float(dataset.true_cate.mean()),
        ate_lower=float(dataset.true_cate.mean()) - 0.1,
        ate_upper=float(dataset.true_cate.mean()) + 0.1,
        cate=pd.DataFrame(
            {
                "cate": dataset.true_cate,
                "cate_lower": dataset.true_cate - 0.1,
                "cate_upper": dataset.true_cate + 0.1,
            }
        ),
    )
    metrics = compare_with_truth(perfect, dataset.true_cate)
    assert metrics["cate_rmse"] == pytest.approx(0.0)
    assert metrics["cate_corr"] == pytest.approx(1.0)
    assert metrics["cate_coverage"] == 1.0 and metrics["ate_covered"]
    with pytest.raises(ValueError, match="行数"):
        compare_with_truth(perfect, dataset.true_cate[:-1])


# ---------------------------------------------------------------------------
# 感度分析
# ---------------------------------------------------------------------------
def test_e_value_known_values() -> None:
    assert e_value_from_rr(2.0) == pytest.approx(2 + np.sqrt(2))
    # RR < 1 は 1/RR と同じ
    assert e_value_from_rr(0.5) == pytest.approx(e_value_from_rr(2.0))
    assert e_value_from_rr(1.0) == pytest.approx(1.0)
    assert np.isnan(e_value_from_rr(np.nan)) and np.isnan(e_value_from_rr(-1.0))


def test_e_value_continuous() -> None:
    result = e_value_continuous(0.5, 0.3, 0.7, sd_outcome=1.0)
    assert result["rr"] == pytest.approx(np.exp(0.91 * 0.5))
    assert result["e_value_point"] > result["e_value_ci"] > 1
    # 効果の向きによらない
    negative = e_value_continuous(-0.5, -0.7, -0.3, sd_outcome=1.0)
    assert negative["e_value_point"] == pytest.approx(result["e_value_point"])
    assert negative["e_value_ci"] == pytest.approx(result["e_value_ci"])
    # CI が 0 を含めば CI の E-value は 1
    assert e_value_continuous(0.1, -0.1, 0.3, sd_outcome=1.0)["e_value_ci"] == 1.0
    # 処置の変化幅を大きくすると E-value も大きい
    assert (
        e_value_continuous(0.5, 0.3, 0.7, 1.0, delta=2.0)["e_value_point"] > result["e_value_point"]
    )
    assert np.isnan(e_value_continuous(np.nan, 0.1, 0.2, 1.0)["e_value_point"])
    with pytest.raises(ValueError):
        e_value_continuous(0.5, 0.3, 0.7, sd_outcome=0.0)

    table = e_value_table(
        pd.DataFrame([{"estimator": "a", "ate": 0.5, "ate_lower": 0.3, "ate_upper": 0.7}]),
        sd_outcome=1.0,
        deltas={"+1": 1.0, "+2": 2.0},
    )
    assert len(table) == 2 and set(table["delta_label"]) == {"+1", "+2"}


def test_e_value_bias_curve_passes_through_e_value() -> None:
    rr = 2.0
    curve = e_value_bias_curve(rr)
    a, b = curve["rr_treatment_confounder"], curve["rr_confounder_outcome"]
    np.testing.assert_allclose(a * b / (a + b - 1), rr, rtol=1e-6)
    # 曲線の上の点で、両方が等しいところが E-value
    closest = (a - b).abs().idxmin()
    assert a[closest] == pytest.approx(e_value_from_rr(rr), rel=0.05)
    assert e_value_bias_curve(1.0).empty


def test_ovb_sensitivity(linear_dml: object) -> None:
    no_confounding = ovb_summary(linear_dml, c_y=0.0, c_t=0.0)
    assert no_confounding["theta_lower"] == pytest.approx(no_confounding["theta"])
    assert no_confounding["theta_upper"] == pytest.approx(no_confounding["theta"])
    assert 0 < no_confounding["rv_ci"] <= no_confounding["rv_theta"] < 1

    grid = ovb_grid(linear_dml, c_values=[0.0, 0.05, 0.1, 0.2])
    assert len(grid) == 16
    # 交絡を強くするほど、範囲は広がる（下限は下がり、上限は上がる）
    for _, group in grid.groupby("c_t"):
        group = group.sort_values("c_y")
        assert group["ci_lower"].is_monotonic_decreasing
        assert group["ci_upper"].is_monotonic_increasing


def test_run_refuters_judges_and_captures_errors() -> None:
    class FakeWrapper:
        estimate_ = SimpleNamespace(value=1.0)

        def refute_estimate(self, method_name: str, **kwargs):
            if method_name == "broken":
                raise RuntimeError("boom")
            new = {"placebo_treatment_refuter": 0.02, "random_common_cause": 0.5}[method_name]
            return SimpleNamespace(
                estimated_effect=1.0, new_effect=new, refutation_result={"p_value": 0.3}
            )

    table = run_refuters(
        FakeWrapper(),
        [
            RefuterSpec("placebo", "placebo_treatment_refuter"),
            RefuterSpec("rcc", "random_common_cause"),
            RefuterSpec("broken", "broken"),
        ],
    ).set_index("refuter")
    assert table.loc["placebo", "judgement"] == "OK"  # 0 に近い
    assert table.loc["rcc", "judgement"] == "要確認"  # 元の推定値から 50% ずれた
    assert table.loc["broken", "error"].startswith("RuntimeError") and np.isnan(
        table.loc["broken", "new_effect"]
    )


def test_dowhy_refuters_run_on_econml_estimator(dataset: SimpleNamespace) -> None:
    rows = slice(0, 400)
    wrapper = fit_dowhy_wrapper(
        _build("linear_dml"),
        dataset.y.iloc[rows],
        dataset.t.iloc[rows],
        dataset.x.iloc[rows],
        dataset.w.iloc[rows],
    )
    specs = [
        s
        for s in default_refuters(num_simulations=2, unobserved_strengths=(0.1,))
        if s.method_name in ("placebo_treatment_refuter", "add_unobserved_common_cause")
    ]
    table = run_refuters(wrapper, specs)
    assert (table["error"] == "").all(), table["error"].tolist()
    assert np.isfinite(table["new_effect"]).all()


# ---------------------------------------------------------------------------
# 図・保存
# ---------------------------------------------------------------------------
def test_plots_and_saver(
    tmp_path: Path, dataset: SimpleNamespace, linear_dml: object, causal_forest: object
) -> None:
    summaries = {
        name: summarize_effects(name, est, dataset.x, extra_alphas=[0.01])
        for name, est in (("LinearDML", linear_dml), ("CausalForestDML", causal_forest))
    }
    cate = {name: s.cate for name, s in summaries.items()}
    ate_table = pd.DataFrame([s.to_row() for s in summaries.values()])
    # 図は欠損・inf の点を除いて描く
    truth_with_nan = dataset.true_cate.copy()
    truth_with_nan[:5] = np.nan

    figures = {
        "ate": plots.plot_ate_comparison(ate_table, true_ate=1.0),
        "scatter": plots.plot_cate_true_vs_estimated(truth_with_nan, cate),
        "dist": plots.plot_cate_distribution(cate, truth_with_nan),
        "by_feature": plots.plot_cate_by_feature(dataset.x, cate, dataset.true_cate),
        "by_feature_single": plots.plot_cate_by_feature(
            dataset.x, {"CausalForestDML": cate["CausalForestDML"]}, features=["X1", "X2", "X3"]
        ),
        "coef": plots.plot_linear_coefficients(
            linear_dml_coefficients(linear_dml, gen.CONFOUNDER_COLS), "係数"
        ),
        "importance": plots.plot_feature_importance(
            pd.Series(causal_forest.feature_importances_, index=gen.CONFOUNDER_COLS), "重要度"
        ),
        "ovb": plots.plot_ovb_contour(
            ovb_grid(linear_dml, c_values=[0.0, 0.1, 0.2, 0.3]), theta=1.0, robustness_value=0.2
        ),
        "e_values": plots.plot_e_values(
            e_value_table(ate_table, sd_outcome=float(dataset.y.std()), deltas={"+1": 1.0})
        ),
        "e_curve": plots.plot_e_value_curve(2.0, e_value_from_rr(2.0), "テスト"),
        "refute": plots.plot_refutations(
            pd.DataFrame(
                [
                    {
                        "refuter": "a",
                        "estimated_effect": 1.0,
                        "new_effect": 0.0,
                        "p_value": 0.5,
                        "judgement": "OK",
                        "error": "",
                    },
                    {
                        "refuter": "b",
                        "estimated_effect": 1.0,
                        "new_effect": np.nan,
                        "p_value": np.nan,
                        "judgement": "-",
                        "error": "x",
                    },
                ]
            )  # fmt: skip
        ),
    }
    saver = OutputSaver.create("dml", run_id="test", root=tmp_path)
    for name, fig in figures.items():
        assert isinstance(fig, plt.Figure)
        saver.figure(fig, name)
    saver.table(ate_table, "ate")
    assert all(path.is_file() for path in saver.paths)
    assert saver.figure_dir == tmp_path / "figures" / "dml" / "test"
    assert (tmp_path / "tables" / "dml" / "test" / "ate.csv").is_file()
    assert len(saver.paths) == len(figures) + 1


def test_cate_by_feature_draws_two_columns_and_shaded_intervals(
    dataset: SimpleNamespace, causal_forest: object
) -> None:
    cate = summarize_effects("CF", causal_forest, dataset.x, extra_alphas=[0.01]).cate
    fig = plots.plot_cate_by_feature(dataset.x, {"CF": cate}, features=["X1", "X2", "X3"])
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    # 1行に2つ（3変数なら 2行 × 2列で、最後の1つは非表示）
    assert len(fig.axes) == 4
    columns = {round(ax.get_position().x0, 3) for ax in visible}
    assert len(columns) == 2
    # 帯は 99%（薄い）→ 95%（濃い）の順に描き、凡例は推定器が1つなら推定器名を付けない
    bands = [c for c in visible[0].collections if c.get_label().endswith("% CI")]
    assert [b.get_label() for b in bands] == ["99% CI", "95% CI"]
    assert bands[0].get_alpha() < bands[1].get_alpha()
    labels = visible[0].get_legend_handles_labels()[1]
    assert "点推定" in labels
    plt.close(fig)

    with pytest.raises(KeyError, match="cate_lower_99"):
        plots.plot_cate_by_feature(
            dataset.x, {"CF": cate[["cate", "cate_lower_95", "cate_upper_95"]]}
        )
    with pytest.raises(ValueError, match="band_alphas"):
        plots.plot_cate_by_feature(dataset.x, {"CF": cate}, band_alphas=[0.3])
