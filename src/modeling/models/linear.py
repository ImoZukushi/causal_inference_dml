"""ベースラインモデル（定数予測）と線形モデル（OLS・LASSO・Ridge・ElasticNet）。

線形モデルは、欠損値補完（中央値）・標準化・モデルを1つにした Pipeline として作る。
前段の特徴量エンジニアリングで欠損を埋めておく必要はない。補完・標準化はどちらも
列ごとの変換なので、SHAP値は係数から厳密に計算できる（`explainer_kind = "linear"`）。
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.models.base import (
    CategoricalEncoding,
    ExplainerKind,
    ModelSpec,
    check_objective,
    merge_params,
    register_model,
    unsupported_objective,
)
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)
_REGRESSION_TASKS = frozenset({Task.REGRESSION, Task.TIME_SERIES})


class WeightedPipeline(Pipeline):
    """`fit(..., sample_weight=...)` を最後のステップ（モデル）に渡す Pipeline。

    scikit-learn の `Pipeline.fit` は `sample_weight` を受け付けない（`ステップ名__sample_weight`
    の形で渡す必要がある）。学習ループはモデルの種類によらず `sample_weight` で重みを渡すため、
    ここで読み替える。欠損値補完・標準化には重みを使わない。
    """

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None, **params: Any) -> Self:
        """前処理を学習し、重みを付けてモデルを学習する。"""
        if sample_weight is not None:
            params[f"{self.steps[-1][0]}__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


def _with_preprocessing(model: BaseEstimator) -> WeightedPipeline:
    """欠損値補完（中央値）+ 標準化 + モデルの Pipeline を作る。"""
    return WeightedPipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


@register_model
class DummySpec(ModelSpec):
    """定数予測（回帰は平均、分類は事前確率）。最低限超えるべきベースライン。"""

    name = "dummy"
    supported_tasks = _ALL_TASKS

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """定数予測モデルを作る。"""
        if is_classification(task):
            return DummyClassifier(**merge_params({"strategy": "prior"}, params))
        return DummyRegressor(**merge_params({"strategy": "mean"}, params))


class _LinearModelSpec(ModelSpec):
    """線形モデルに共通の定義（前処理込みの Pipeline・二乗誤差だけに対応・ワンホット推奨）。"""

    explainer_kind: ClassVar[ExplainerKind] = "linear"
    categorical_encoding: ClassVar[CategoricalEncoding] = "onehot"

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """目的関数は二乗誤差（`l2`）だけに対応する（追加のパラメータは無い）。

        Raises:
            ValueError: 未知の名前、または `l2` 以外の場合。
        """
        check_objective(objective)
        if objective != "l2":
            raise unsupported_objective(self, objective)
        return {}


@register_model
class LinearSpec(_LinearModelSpec):
    """線形モデル（回帰はRidge、分類はLogisticRegression）。

    欠損値補完（中央値）と標準化を内部に含めたPipelineを返すため、
    前段の特徴量エンジニアリングで欠損を埋めておく必要はない。
    """

    name = "linear"
    supported_tasks = _ALL_TASKS
    # 既存の結果との互換のため、SHAPは従来どおり permutation 法で計算する
    explainer_kind = "permutation"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """欠損補完 + 標準化 + 線形モデルのPipelineを作る。"""
        model: BaseEstimator
        if is_classification(task):
            model = LogisticRegression(
                **merge_params({"max_iter": 1000, "random_state": seed}, params)
            )
        else:
            model = Ridge(**merge_params({"alpha": 1.0}, params))
        return _with_preprocessing(model)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """正則化の強さを対数スケールで探索する。"""
        if is_classification(task):
            return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


class _LinearRegressorSpec(_LinearModelSpec):
    """回帰専用の線形モデル（`estimator` に既定のパラメータで渡すクラスを指定する）。"""

    supported_tasks = _REGRESSION_TASKS
    estimator: ClassVar[type[BaseEstimator]]
    defaults: ClassVar[dict[str, Any]] = {}
    # 乱数シードを受け取るモデル（座標降下法で selection="random" のとき使う）
    uses_seed: ClassVar[bool] = False

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """欠損補完 + 標準化 + 線形回帰のPipelineを作る（early stoppingは無い）。

        Raises:
            ValueError: 分類タスクの場合。
        """
        if is_classification(task):
            raise ValueError(f"{self.name} は回帰専用のモデルです（タスク: {task.value}）")
        defaults = dict(self.defaults)
        if self.uses_seed:
            defaults["random_state"] = seed
        return _with_preprocessing(self.estimator(**merge_params(defaults, params)))


@register_model
class OLSSpec(_LinearRegressorSpec):
    """最小二乗法（`LinearRegression`。正則化なし）。

    ハイパーパラメータが無いため、Optunaの探索空間は持たない。説明変数どうしが強く相関する
    （ワンホットの全列と切片など）と係数は一意に決まらないが、予測値は最小ノルム解で求まる。
    """

    name = "ols"
    estimator = LinearRegression


@register_model
class LassoSpec(_LinearRegressorSpec):
    """LASSO（`Lasso`。L1正則化。係数の一部がちょうど0になり、特徴量を選ぶ働きがある）。"""

    name = "lasso"
    estimator = Lasso
    # 正則化が弱いと座標降下法の収束に時間がかかるため、反復の上限を既定（1000）より増やす
    defaults: ClassVar[dict[str, Any]] = {"alpha": 1.0, "max_iter": 10000}
    uses_seed = True

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """正則化の強さ（`alpha`）を対数スケールで探索する。"""
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}


@register_model
class RidgeSpec(_LinearRegressorSpec):
    """Ridge（`Ridge`。L2正則化。係数を全体に小さくし、相関の強い説明変数でも安定する）。

    回帰の `linear` と同じモデル。`linear` との違いは、SHAPを係数から厳密に計算する点。
    """

    name = "ridge"
    estimator = Ridge
    defaults: ClassVar[dict[str, Any]] = {"alpha": 1.0}

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """正則化の強さ（`alpha`）を対数スケールで探索する。"""
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


@register_model
class ElasticNetSpec(_LinearRegressorSpec):
    """ElasticNet（`ElasticNet`。L1とL2の正則化の組み合わせ。`l1_ratio=1` でLASSOと同じ）。"""

    name = "elastic_net"
    estimator = ElasticNet
    defaults: ClassVar[dict[str, Any]] = {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 10000}
    uses_seed = True

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """正則化の強さ（`alpha`）と、L1の割合（`l1_ratio`）を探索する。"""
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0),
        }
