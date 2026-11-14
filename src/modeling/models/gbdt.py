"""勾配ブースティング決定木（LightGBM・XGBoost・CatBoost）。"""

from __future__ import annotations

from typing import Any

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator

from evaluation.curves import TrainingHistory
from modeling.models.base import (
    ModelSpec,
    check_objective,
    merge_params,
    objective_alpha,
    register_model,
)
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)


@register_model
class LightGBMSpec(ModelSpec):
    """LightGBM（`LGBMRegressor` / `LGBMClassifier`）。"""

    name = "lightgbm"
    supported_tasks = _ALL_TASKS
    explainer_kind = "tree"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """LightGBMのestimatorを作る（early stoppingは `fit_kwargs` のcallbackで指定）。"""
        merged = merge_params({"random_state": seed, "verbose": -1}, params)
        if is_classification(task):
            return lgb.LGBMClassifier(**merged)
        return lgb.LGBMRegressor(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """LightGBMの代表的なパラメータの探索空間。"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def fit_kwargs(
        self,
        X_valid: Any,
        y_valid: Any,
        early_stopping_rounds: int | None,
        *,
        X_train: Any = None,
        y_train: Any = None,
        record_history: bool = False,
    ) -> dict[str, Any]:
        """early stopping・学習の推移の記録用に、評価セットとcallbackを渡す。

        学習データを評価セットに加える場合は、`fit` に渡すのと同じオブジェクトを渡す
        （LightGBMは同一オブジェクトを学習データとみなし、early stoppingの判定から除く）。
        """
        eval_X: list[Any] = []
        eval_y: list[Any] = []
        if record_history and X_train is not None:
            eval_X.append(X_train)
            eval_y.append(y_train)
        if X_valid is not None and (early_stopping_rounds is not None or record_history):
            eval_X.append(X_valid)
            eval_y.append(y_valid)
        kwargs: dict[str, Any] = {}
        if eval_X:
            # LightGBM 4.7以降は eval_set ではなく eval_X / eval_y で渡す
            kwargs.update(eval_X=tuple(eval_X), eval_y=tuple(eval_y))
        if early_stopping_rounds is not None and X_valid is not None:
            kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        return kwargs

    def training_history(self, estimator: Any) -> TrainingHistory | None:
        """`evals_result_` から学習（`training`）・検証（それ以外）の損失を取り出す。"""
        results = getattr(estimator, "evals_result_", None) or {}
        if not results:
            return None
        train = results.get("training")
        valid = next((v for k, v in results.items() if k != "training"), None)
        metric = next(iter((train or valid or {}).keys()), "loss")
        return TrainingHistory(
            metric=metric,
            train=list(train[metric]) if train else None,
            valid=list(valid[metric]) if valid else None,
            best_iteration=self.best_iteration(estimator),
        )

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数。"""
        best = getattr(estimator, "best_iteration_", None)
        return int(best) if best else None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """木の本数を固定したパラメータを返す。"""
        return {**params, "n_estimators": n_iterations}

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """目的関数を LightGBM の `objective`（huber・quantile は `alpha` も）に変換する。"""
        check_objective(objective)
        names = {"l2": "regression", "l1": "regression_l1", "poisson": "poisson"}
        if objective in names:
            return {"objective": names[objective]}
        return {"objective": objective, "alpha": objective_alpha(objective, alpha)}


@register_model
class XGBoostSpec(ModelSpec):
    """XGBoost（`XGBRegressor` / `XGBClassifier`）。"""

    name = "xgboost"
    supported_tasks = _ALL_TASKS
    explainer_kind = "tree"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """XGBoostのestimatorを作る（XGBoost 2.0以降はearly stoppingをコンストラクタで指定）。"""
        defaults: dict[str, Any] = {"random_state": seed, "tree_method": "hist"}
        if early_stopping_rounds is not None:
            defaults["early_stopping_rounds"] = early_stopping_rounds
        merged = merge_params(defaults, params)
        if is_classification(task):
            return xgb.XGBClassifier(**merged)
        return xgb.XGBRegressor(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """XGBoostの代表的なパラメータの探索空間。"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def fit_kwargs(
        self,
        X_valid: Any,
        y_valid: Any,
        early_stopping_rounds: int | None,
        *,
        X_train: Any = None,
        y_train: Any = None,
        record_history: bool = False,
    ) -> dict[str, Any]:
        """early stopping・学習の推移の記録用に評価セットを渡す。

        XGBoostのearly stoppingは最後の評価セットで判定するため、学習データを先、
        検証データを最後に並べる。
        """
        eval_set: list[tuple[Any, Any]] = []
        if record_history and X_train is not None:
            eval_set.append((X_train, y_train))
        if X_valid is not None and (early_stopping_rounds is not None or record_history):
            eval_set.append((X_valid, y_valid))
        if not eval_set:
            return {}
        return {"eval_set": eval_set, "verbose": False}

    def training_history(self, estimator: Any) -> TrainingHistory | None:
        """`evals_result()` から学習・検証の損失を取り出す。

        学習の推移を記録した場合（`record_history=True`）は、評価セットが2つなら
        学習・検証の順、1つなら学習データのみ（検証データが無い全データ再学習など）。
        """
        try:
            results = estimator.evals_result()
        except xgb.core.XGBoostError:
            # 評価セット無しで学習した場合、XGBoostは例外を送出する
            return None
        if not results:
            return None
        sets = list(results.values())
        metric = next(iter(sets[0].keys()), "loss")
        train = list(sets[0][metric])
        valid = list(sets[1][metric]) if len(sets) > 1 else None
        return TrainingHistory(metric, train, valid, self.best_iteration(estimator))

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数（0始まりのため+1する）。"""
        try:
            return int(estimator.best_iteration) + 1
        except AttributeError:
            # early stoppingなしで学習した場合は best_iteration が存在しない
            return None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """木の本数を固定したパラメータを返す。"""
        return {**params, "n_estimators": n_iterations}

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """目的関数を XGBoost の `objective` に変換する。

        huber は `huber_slope`、quantile は `quantile_alpha` に `alpha` を渡す。
        XGBoost の huber は擬似Huber損失（Huber損失の滑らかな近似）。
        """
        check_objective(objective)
        names = {"l2": "reg:squarederror", "l1": "reg:absoluteerror", "poisson": "count:poisson"}
        if objective in names:
            return {"objective": names[objective]}
        value = objective_alpha(objective, alpha)
        if objective == "huber":
            return {"objective": "reg:pseudohubererror", "huber_slope": value}
        return {"objective": "reg:quantileerror", "quantile_alpha": value}


@register_model
class CatBoostSpec(ModelSpec):
    """CatBoost（`CatBoostRegressor` / `CatBoostClassifier`）。

    木の本数は `iterations`（`n_estimators` を渡した場合は `iterations` に読み替える）。
    学習データの損失は、評価セットに加えなくても毎反復記録される。
    """

    name = "catboost"
    supported_tasks = _ALL_TASKS
    explainer_kind = "tree"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """CatBoostのestimatorを作る（early stoppingは `fit_kwargs` で指定）。"""
        params = dict(params)
        # 同じ意味の別名を両方渡すとエラーになるため、他のモデルと共通の名前を読み替える
        n_estimators = params.pop("n_estimators", None)
        if n_estimators is not None:
            params.setdefault("iterations", n_estimators)
        defaults: dict[str, Any] = {
            "verbose": False,
            # 学習ログのディレクトリ（catboost_info/）を作らない
            "allow_writing_files": False,
        }
        if "random_state" not in params:
            defaults["random_seed"] = seed
        merged = merge_params(defaults, params)
        if is_classification(task):
            return cb.CatBoostClassifier(**merged)
        return cb.CatBoostRegressor(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """CatBoostの代表的なパラメータの探索空間。

        深さの上限は8にする（CatBoost の木は左右対称で葉が 2^深さ 個になり、深さ10では
        1万行・1000本の学習に1 fold あたり数分かかるため）。
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }

    def fit_kwargs(
        self,
        X_valid: Any,
        y_valid: Any,
        early_stopping_rounds: int | None,
        *,
        X_train: Any = None,
        y_train: Any = None,
        record_history: bool = False,
    ) -> dict[str, Any]:
        """early stopping・学習の推移の記録用に、検証データを評価セットとして渡す。

        CatBoost は評価セットを渡すと、既定で最良の反復まで木を切り詰める（`use_best_model`）。
        early stopping しない場合は明示的に無効にし、学習の推移を記録するだけで
        結果が変わらないようにする。
        """
        if X_valid is None or (early_stopping_rounds is None and not record_history):
            return {}
        kwargs: dict[str, Any] = {"eval_set": [(X_valid, y_valid)], "use_best_model": False}
        if early_stopping_rounds is not None:
            kwargs.update(early_stopping_rounds=early_stopping_rounds, use_best_model=True)
        return kwargs

    def training_history(self, estimator: Any) -> TrainingHistory | None:
        """`get_evals_result()` から学習（`learn`）・検証（それ以外）の損失を取り出す。"""
        try:
            results = estimator.get_evals_result()
        except cb.CatBoostError:
            return None
        if not results:
            return None
        train = results.get("learn")
        valid = next((v for k, v in results.items() if k != "learn"), None)
        metric = next(iter((train or valid or {}).keys()), "loss")
        return TrainingHistory(
            metric=metric,
            train=list(train[metric]) if train and metric in train else None,
            valid=list(valid[metric]) if valid and metric in valid else None,
            best_iteration=self.best_iteration(estimator),
        )

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数（0始まりのため+1する）。"""
        try:
            best = estimator.get_best_iteration()
        except cb.CatBoostError:
            return None
        return int(best) + 1 if best is not None else None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """木の本数を固定したパラメータを返す（`n_estimators` は `iterations` に読み替える）。"""
        rest = {k: v for k, v in params.items() if k != "n_estimators"}
        return {**rest, "iterations": n_iterations}

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """目的関数を CatBoost の `loss_function` に変換する。

        huber は `delta`、quantile は `alpha` として損失の名前に埋め込む。
        """
        check_objective(objective)
        names = {"l2": "RMSE", "l1": "MAE", "poisson": "Poisson"}
        if objective in names:
            return {"loss_function": names[objective]}
        value = objective_alpha(objective, alpha)
        if objective == "huber":
            return {"loss_function": f"Huber:delta={value}"}
        return {"loss_function": f"Quantile:alpha={value}"}
