"""線形モデル（OLS・LASSO・Ridge・ElasticNet）と、線形モデルのSHAPのテスト。"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from modeling.config import ExperimentConfig
from modeling.experiment import prepare_dataset
from modeling.explain import compute_oof_shap
from modeling.models import get_model_spec
from modeling.tasks import Task
from modeling.trainer import fit_pipeline, run_cv

LINEAR_MODELS = ["ols", "lasso", "ridge", "elastic_net"]


def _config(model: str, params: dict[str, Any] | None = None, **raw: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "lin",
            "task": "regression",
            "data": {"train_path": "unused.csv", "target": "y"},
            "cv": {"method": "kfold", "n_splits": 3},
            "model": {"name": model, "params": params or {}},
            "metrics": ["mae"],
            **raw,
        }
    )


def _frame(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = 3 * X["b"].to_numpy() - 2 * X["a"].to_numpy() + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize("model", LINEAR_MODELS)
def test_linear_models_fit_with_nan_inf_and_learn_signal(model: str) -> None:
    X, y = _frame()
    X.loc[::5, "a"] = np.nan
    X.loc[1::7, "c"] = np.inf
    params = {"alpha": 0.01} if model in ("lasso", "elastic_net") else {}
    pipeline = fit_pipeline(_config(model, params), X, y, sample_weight=np.ones(len(X)))
    pred = pipeline.predict(X)
    assert np.isfinite(pred).all()
    observed = X["a"].notna().to_numpy()
    assert np.abs(pred - y)[observed].mean() < 0.3


@pytest.mark.parametrize("model", ["linear", *LINEAR_MODELS])
def test_linear_models_use_sample_weight(model: str) -> None:
    # 特徴量はグループと無関係。片方のグループに大きな重みを付けると、予測がそちらに寄る
    n = 200
    X = pd.DataFrame({"x": (np.arange(n) % 2).astype(float)})
    y = np.where(np.arange(n) < n // 2, 0.0, 10.0)
    params = {"alpha": 1e-3} if model != "ols" else {}
    plain = fit_pipeline(_config(model, params), X, y).predict(X).mean()
    weights = np.where(y > 0, 9.0, 1.0)
    weighted = fit_pipeline(_config(model, params), X, y, sample_weight=weights).predict(X).mean()
    assert plain == pytest.approx(5.0, abs=0.1)
    assert weighted == pytest.approx(9.0, abs=0.1)


def test_lasso_sets_noise_coefficients_to_zero() -> None:
    X, y = _frame()
    pipeline = fit_pipeline(_config("lasso", {"alpha": 0.1}), X, y)
    coef = pipeline.named_steps["model"][-1].coef_
    assert coef[2] == 0.0  # 目的変数と無関係な c
    assert coef[0] != 0.0 and coef[1] != 0.0


@pytest.mark.parametrize("model", LINEAR_MODELS)
def test_linear_models_only_support_l2_objective(model: str) -> None:
    spec = get_model_spec(model)
    assert spec.objective_params("l2") == {}
    for objective in ("l1", "huber", "quantile", "poisson"):
        with pytest.raises(ValueError, match="対応していません"):
            spec.objective_params(objective)


@pytest.mark.parametrize("model", LINEAR_MODELS)
def test_linear_models_are_regression_only_and_prefer_onehot(model: str) -> None:
    spec = get_model_spec(model)
    assert spec.supports(Task.REGRESSION)
    assert not spec.supports(Task.BINARY)
    assert spec.categorical_encoding == "onehot"
    assert spec.explainer_kind == "linear"
    with pytest.raises(ValueError, match="回帰専用"):
        spec.build(Task.BINARY, {}, seed=0)


def test_tree_models_prefer_ordinal_encoding() -> None:
    assert get_model_spec("lightgbm").categorical_encoding == "ordinal"


@pytest.mark.parametrize("model", ["lasso", "ridge", "elastic_net"])
def test_linear_search_space_samples_buildable_params(model: str) -> None:
    trial = optuna.create_study().ask()
    params = get_model_spec(model).search_space(trial, Task.REGRESSION)
    assert params
    X, y = _frame(n=80)
    assert np.isfinite(fit_pipeline(_config(model, params), X, y).predict(X)).all()


def test_ols_has_no_search_space() -> None:
    assert not get_model_spec("ols").has_search_space
    assert get_model_spec("lasso").has_search_space
    assert get_model_spec("lightgbm").has_search_space
    with pytest.raises(NotImplementedError):
        get_model_spec("ols").search_space(optuna.create_study().ask(), Task.REGRESSION)


@pytest.mark.parametrize("model", LINEAR_MODELS)
def test_linear_shap_is_exact_and_additive(synthetic_frame: pd.DataFrame, model: str) -> None:
    frame = synthetic_frame.copy()
    frame.loc[frame.index[::6], "a"] = np.nan  # 欠損は中央値で補完される
    cfg = _config(
        model,
        {"alpha": 0.01} if model in ("lasso", "elastic_net") else {},
        data={
            "train_path": "unused.csv",
            "target": "y_reg",
            "drop_cols": ["id", "g", "ts", "y_bin", "y_multi"],
        },
        features=[
            {
                "class": "feature_engineering.categorical.PandasOneHotEncoder",
                "params": {"variables": ["cat"]},
            },
            {"class": "modeling.pipeline.DropColumns", "params": {"columns": ["cat"]}},
        ],
    )
    ds = prepare_dataset(cfg, frame)
    result = run_cv(cfg, ds.X, ds.y, ds.folds)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result)
    assert shap_result.explainer_kind == "linear"
    assert any(c.startswith("cat_") for c in shap_result.feature_names)
    reconstructed = shap_result.values.sum(axis=1) + shap_result.base_values
    assert reconstructed == pytest.approx(result.oof_pred[shap_result.rows], abs=1e-8)
