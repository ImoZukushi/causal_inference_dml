"""ランダムフォレスト系のモデル（RandomForest・ExtraTrees。scikit-learn）。

どちらも木を独立に育てて平均するため、early stopping・学習の推移（反復ごとの損失）は無い。
欠損（NaN）はそのまま扱える（scikit-learn 1.4以降）。±inf は扱えないが、
`modeling.pipeline.ToModelInput` がモデルに渡す前に NaN にする。
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from modeling.models.base import (
    ModelSpec,
    check_objective,
    merge_params,
    register_model,
    unsupported_objective,
)
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)

# 目的関数の共通の名前 → 回帰木の分割基準（criterion）。huber・quantile は対応しない
_CRITERIA = {"l2": "squared_error", "l1": "absolute_error", "poisson": "poisson"}


class _ForestSpec(ModelSpec):
    """RandomForest・ExtraTrees に共通の定義。"""

    supported_tasks = _ALL_TASKS
    explainer_kind = "tree"
    # 深い木を多数持つため TreeSHAP が1行あたり0.2〜0.4秒かかる
    # （約1万行・57列・葉の最小5行・300本で計測）。既定の2000行では10分を超えるので、行数を抑える
    shap_max_samples = 300
    regressor: ClassVar[type[BaseEstimator]]
    classifier: ClassVar[type[BaseEstimator]]

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """estimatorを作る（early stoppingは無いため `early_stopping_rounds` は使わない）。

        既定の `min_samples_leaf=5` は回帰のランダムフォレストでよく使われる値
        （scikit-learn の既定の1では葉が1行ずつになるまで木が育ち、予測・SHAP が重くなる）。
        """
        defaults = {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1, "random_state": seed}
        merged = merge_params(defaults, params)
        estimator_cls = self.classifier if is_classification(task) else self.regressor
        return estimator_cls(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """木の深さ・葉の最小サンプル数・分割に使う特徴量の割合の探索空間。

        ランダムフォレスト系は深い木を平均して分散を下げるので、浅い木（深さ8未満）や
        特徴量の割合が小さすぎる設定は探さない（大きく学習不足になる）。葉の最小サンプル数は
        3以上にする（1・2では葉が増え、予測と TreeSHAP がとても重くなる）。
        """
        return {
            "max_depth": trial.suggest_int("max_depth", 8, 48, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """目的関数を分割基準（`criterion`）に変換する。

        `l1`（absolute_error）は分割の評価が重く、`l2` よりかなり遅い。

        Raises:
            ValueError: 未知の名前、または huber・quantile の場合（scikit-learn の木に無い）。
        """
        check_objective(objective)
        if objective not in _CRITERIA:
            raise unsupported_objective(self, objective)
        return {"criterion": _CRITERIA[objective]}


@register_model
class RandomForestSpec(_ForestSpec):
    """ランダムフォレスト（`RandomForestRegressor` / `RandomForestClassifier`）。"""

    name = "random_forest"
    regressor = RandomForestRegressor
    classifier = RandomForestClassifier

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """共通の探索空間に、木ごとに使う行の割合（`max_samples`）を加える。"""
        return {
            **super().search_space(trial, task),
            "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
        }


@register_model
class ExtraTreesSpec(_ForestSpec):
    """ExtraTrees（`ExtraTreesRegressor` / `ExtraTreesClassifier`）。

    分割の閾値をランダムに選ぶため、RandomForest より速く、分散が小さくなりやすい。
    """

    name = "extra_trees"
    regressor = ExtraTreesRegressor
    classifier = ExtraTreesClassifier
