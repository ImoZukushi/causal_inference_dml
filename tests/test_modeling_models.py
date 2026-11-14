"""モデル定義（`ModelSpec`）のテスト（目的関数の対応・欠損や ±inf の扱い・CatBoost 固有の処理）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from modeling.config import ExperimentConfig
from modeling.models import get_model_spec
from modeling.models.base import OBJECTIVES
from modeling.pipeline import ToModelInput
from modeling.tasks import Task
from modeling.trainer import fit_pipeline

TREE_MODELS = ["lightgbm", "xgboost", "catboost", "random_forest", "extra_trees"]


def _config(
    model: str, params: dict[str, Any] | None = None, **model_opts: Any
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "m",
            "task": "regression",
            "data": {"train_path": "unused.csv", "target": "y"},
            "cv": {"method": "kfold", "n_splits": 2},
            "model": {"name": model, "params": params or {"n_estimators": 20}, **model_opts},
            "metrics": ["mae"],
        }
    )


def _frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = 3 * X["b"].to_numpy() + rng.normal(scale=0.1, size=n)
    return X, y


# --- ToModelInput ---------------------------------------------------------------------


def test_to_model_input_replaces_inf_with_nan() -> None:
    frame = pd.DataFrame({"x": [1.0, np.inf, -np.inf, np.nan], "k": [1, 2, 3, 4]})
    out = ToModelInput().fit(frame).transform(frame)
    assert out["x"].tolist()[0] == 1.0
    assert out["x"].isna().tolist() == [False, True, True, True]
    assert out["k"].tolist() == [1, 2, 3, 4]
    # 入力は書き換えない
    assert np.isinf(frame["x"]).sum() == 2


# --- 学習・予測 -----------------------------------------------------------------------


@pytest.mark.parametrize("model", TREE_MODELS)
def test_models_fit_and_predict_with_nan_and_inf_features(model: str) -> None:
    X, y = _frame()
    X.loc[::5, "a"] = np.nan
    X.loc[1::7, "c"] = np.inf
    X.loc[2::11, "c"] = -np.inf
    X_valid = X.iloc[:40]
    pipeline = fit_pipeline(_config(model), X, y, sample_weight=np.linspace(1.0, 2.0, len(X)))
    pred = pipeline.predict(X_valid)
    assert pred.shape == (40,)
    assert np.isfinite(pred).all()
    # 信号のある列 b を学習している（定数予測より十分よい）
    assert np.abs(pred - y[:40]).mean() < np.abs(y[:40] - y.mean()).mean() / 2


# --- 目的関数 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "objective", "alpha", "expected"),
    [
        ("lightgbm", "l2", None, {"objective": "regression"}),
        ("lightgbm", "l1", None, {"objective": "regression_l1"}),
        ("lightgbm", "huber", 2.0, {"objective": "huber", "alpha": 2.0}),
        ("lightgbm", "quantile", None, {"objective": "quantile", "alpha": 0.5}),
        ("lightgbm", "poisson", None, {"objective": "poisson"}),
        ("xgboost", "l2", None, {"objective": "reg:squarederror"}),
        ("xgboost", "l1", None, {"objective": "reg:absoluteerror"}),
        ("xgboost", "huber", None, {"objective": "reg:pseudohubererror", "huber_slope": 1.0}),
        ("xgboost", "quantile", 0.9, {"objective": "reg:quantileerror", "quantile_alpha": 0.9}),
        ("xgboost", "poisson", None, {"objective": "count:poisson"}),
        ("catboost", "l2", None, {"loss_function": "RMSE"}),
        ("catboost", "l1", None, {"loss_function": "MAE"}),
        ("catboost", "huber", 2.0, {"loss_function": "Huber:delta=2.0"}),
        ("catboost", "quantile", None, {"loss_function": "Quantile:alpha=0.5"}),
        ("catboost", "poisson", None, {"loss_function": "Poisson"}),
        ("random_forest", "l2", None, {"criterion": "squared_error"}),
        ("random_forest", "l1", None, {"criterion": "absolute_error"}),
        ("extra_trees", "poisson", None, {"criterion": "poisson"}),
    ],
)
def test_objective_params_maps_common_names(
    model: str, objective: str, alpha: float | None, expected: dict[str, Any]
) -> None:
    assert get_model_spec(model).objective_params(objective, alpha) == expected


@pytest.mark.parametrize("model", ["random_forest", "extra_trees"])
@pytest.mark.parametrize("objective", ["huber", "quantile"])
def test_forest_rejects_unsupported_objectives(model: str, objective: str) -> None:
    with pytest.raises(ValueError, match="対応していません"):
        get_model_spec(model).objective_params(objective)


@pytest.mark.parametrize(
    ("objective", "alpha"), [("no_such", None), ("quantile", 1.5), ("huber", 0.0)]
)
def test_objective_params_rejects_invalid_input(objective: str, alpha: float | None) -> None:
    with pytest.raises(ValueError):
        get_model_spec("lightgbm").objective_params(objective, alpha)


def test_models_without_mapping_reject_objectives() -> None:
    with pytest.raises(ValueError, match="対応していません"):
        get_model_spec("dummy").objective_params("l2")


@pytest.mark.parametrize("model", TREE_MODELS)
def test_every_supported_objective_trains(model: str) -> None:
    X, y = _frame(n=120)
    # poisson は非負の目的変数が必要
    y = np.abs(y)
    spec = get_model_spec(model)
    for objective in OBJECTIVES:
        try:
            params = spec.objective_params(objective)
        except ValueError:
            continue
        cfg = _config(model, {"n_estimators": 10, **params})
        pred = fit_pipeline(cfg, X, y).predict(X)
        assert np.isfinite(pred).all(), objective


# --- CatBoost -------------------------------------------------------------------------


def test_catboost_early_stopping_and_training_history() -> None:
    X, y = _frame(n=300)
    spec = get_model_spec("catboost")
    cfg = _config(
        "catboost", {"n_estimators": 2000, "learning_rate": 0.3}, early_stopping_rounds=10
    )
    pipeline = fit_pipeline(cfg, X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    model = pipeline.named_steps["model"]
    best = spec.best_iteration(model)
    assert best is not None and 1 <= best < 2000
    assert model.tree_count_ == best  # use_best_model で最良の反復まで切り詰める
    history = spec.training_history(model)
    assert history is not None
    assert history.metric == "RMSE"
    assert history.train is not None and history.valid is not None
    assert history.best_iteration == best


def test_catboost_without_valid_has_no_best_iteration() -> None:
    X, y = _frame(n=100)
    spec = get_model_spec("catboost")
    model = fit_pipeline(_config("catboost"), X, y).named_steps["model"]
    assert spec.best_iteration(model) is None
    assert model.tree_count_ == 20  # n_estimators が iterations に読み替えられる


def test_catboost_with_n_iterations_uses_iterations() -> None:
    spec = get_model_spec("catboost")
    params = spec.with_n_iterations({"n_estimators": 1000, "depth": 4}, 37)
    assert params == {"depth": 4, "iterations": 37}
    model = spec.build(Task.REGRESSION, params, seed=0)
    assert model.get_params()["iterations"] == 37


def test_catboost_record_history_without_early_stopping_keeps_all_trees() -> None:
    X, y = _frame(n=200)
    spec = get_model_spec("catboost")
    kwargs = spec.fit_kwargs(X.iloc[150:], y[150:], None, record_history=True)
    assert kwargs["use_best_model"] is False
    model = spec.build(Task.REGRESSION, {"iterations": 30}, seed=0)
    model.fit(X.iloc[:150], y[:150], **kwargs)
    assert model.tree_count_ == 30


# --- 探索空間 -------------------------------------------------------------------------


@pytest.mark.parametrize("model", TREE_MODELS)
def test_search_space_samples_buildable_params(model: str) -> None:
    spec = get_model_spec(model)
    trial = optuna.create_study().ask()
    params = spec.search_space(trial, Task.REGRESSION)
    assert params
    X, y = _frame(n=80)
    pred = fit_pipeline(_config(model, {"n_estimators": 5, **params}), X, y).predict(X)
    assert np.isfinite(pred).all()
