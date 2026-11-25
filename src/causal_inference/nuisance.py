"""cross-fitting の分割と、nuisance モデルの Optuna チューニング。

DML の nuisance モデルは、結果のモデル E[Y|X,W]（`model_y`）と処置のモデル E[T|X,W]（`model_t`）。
econml は両者に `[X, W]` を横に並べた行列を渡すので、チューニングも同じ並びの特徴量で行う。

cross-fitting の分割は、行が独立なら K-fold（シャッフルあり）、時系列なら時間順の連続ブロック
（`BlockedTimeSeriesSplit`。検証ブロックの前後 `gap` 行を学習から除く）から選ぶ（`make_cv`）。

チューニングの目的関数は、cross-fitting と同じ分割方法・分割数での out-of-fold MSE（fold 平均）。
fold ごとのスコアを `trial.report` に渡し、見込みの薄い試行は MedianPruner で打ち切る。

注意: ハイパーパラメータは全データの CV で選ぶため、選んだ値は推定に使うデータを一度見ている。
nuisance の予測精度を選ぶだけで処置効果は直接見ていないので、通常は問題にならない。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from modeling.models import get_model_spec
from modeling.tasks import Task

# 試行ごとの INFO ログを抑える（結果は NuisanceTuningResult.trials で確認する）
optuna.logging.set_verbosity(optuna.logging.WARNING)

NUISANCE_MODEL = "lightgbm"
# チューニングしないときの LightGBM のパラメータ（データ数が数千〜数万の想定で控えめな値）
DEFAULT_NUISANCE_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    # 既定（全スレッド）だと、コア数の多い環境で小さなデータを学習するときにスレッドの取り合いで
    # 大幅に遅くなる（16スレッドの Windows で 5 倍以上）ので、少なめに固定する
    "n_jobs": 4,
}

SearchSpace = dict[str, dict[str, Any]] | Callable[[optuna.Trial], dict[str, Any]]


def check_finite(name: str, values: np.ndarray | pd.DataFrame | pd.Series | None) -> None:
    """配列に欠損・inf が無いことを確かめる（econml・DoWhy に渡す前の検査）。

    Args:
        name: エラー文に出す変数名。
        values: 検査する値（None なら何もしない）。

    Raises:
        ValueError: 欠損または inf を含む場合。
    """
    if values is None:
        return
    array = np.asarray(values, dtype=float)
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(
            f"{name} に欠損または inf が {int(bad.sum())} 個あります。"
            "推定の前に除外または補完してください。"
        )


CV_STRATEGIES = ("kfold", "timeseries")


class BlockedTimeSeriesSplit:
    """時系列の cross-fitting 用の分割（時間順の連続ブロック + 前後 `gap` 行の除外）。

    行が時間順に並んでいる前提で、全行を `n_splits` 個の連続したブロックに分け、各ブロックを順に
    検証に使う。学習には検証ブロック以外のすべての行（過去と未来）を使うが、検証ブロックの
    前後 `gap` 行は除く（neighbors-left-out。自己相関によって検証ブロックの情報が学習に漏れるのを
    抑える）。

    過去だけで学習する前向きの分割（`sklearn.model_selection.TimeSeriesSplit`）は、最初のブロックが
    どの検証にも入らず、その行の nuisance の予測が作れないため、DML の cross-fitting には使えない。

    分割は乱数を使わず決まるので、`mc_iters` で繰り返しても同じ分割になる。

    Args:
        n_splits: ブロックの数（2以上）。
        gap: 検証ブロックの前後で学習から除く行数（0以上）。
    """

    def __init__(self, n_splits: int = 5, gap: int = 0) -> None:
        if n_splits < 2:
            raise ValueError(f"cross-fitting には n_splits >= 2 が必要です: {n_splits}")
        if gap < 0:
            raise ValueError(f"gap は0以上にしてください: {gap}")
        self.n_splits = n_splits
        self.gap = gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """分割数を返す（sklearn の分割器と同じ形）。"""
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        """(学習の行番号, 検証の行番号) を順に返す。

        Raises:
            ValueError: 行数がブロック数より少ない、または `gap` が大きすぎて学習の行が無い場合。
        """
        n_rows = len(X)
        if n_rows < self.n_splits:
            raise ValueError(f"行数 {n_rows} が分割数 {self.n_splits} より少ないです")
        indices = np.arange(n_rows)
        for block in np.array_split(indices, self.n_splits):
            start, stop = int(block[0]), int(block[-1]) + 1
            train = indices[(indices < start - self.gap) | (indices >= stop + self.gap)]
            if len(train) == 0:
                raise ValueError(f"gap={self.gap} が大きすぎて、学習に使える行がありません")
            yield train, block

    def __repr__(self) -> str:
        return f"BlockedTimeSeriesSplit(n_splits={self.n_splits}, gap={self.gap})"


type Splitter = KFold | BlockedTimeSeriesSplit


def make_cv(n_splits: int, seed: int = 0, strategy: str = "kfold", gap: int = 0) -> Splitter:
    """cross-fitting に使う分割器を作る。

    Args:
        n_splits: 分割数（2以上）。
        seed: シャッフルの乱数シード（`kfold` のとき）。
        strategy: "kfold"（行をシャッフルした K-fold。行が独立な場合）か "timeseries"
            （時間順の連続ブロック。行が時間順に並んだ時系列の場合。`BlockedTimeSeriesSplit`）。
        gap: 検証ブロックの前後で学習から除く行数（`timeseries` のとき）。

    Returns:
        分割器。

    Raises:
        ValueError: `n_splits` が2未満（cross-fitting ができない）、未知の `strategy`、
            または `kfold` で `gap` を指定した場合。
    """
    if n_splits < 2:
        raise ValueError(f"cross-fitting には n_splits >= 2 が必要です: {n_splits}")
    if strategy == "kfold":
        if gap:
            raise ValueError("gap は strategy='timeseries' のときだけ指定できます")
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "timeseries":
        return BlockedTimeSeriesSplit(n_splits=n_splits, gap=gap)
    raise ValueError(f"未知の分割方法です: {strategy}（{CV_STRATEGIES} のいずれか）")


def make_nuisance_model(params: dict[str, Any] | None = None, seed: int = 0) -> BaseEstimator:
    """nuisance モデル（LightGBM の回帰）を作る。

    Args:
        params: LightGBM のパラメータ（None なら `DEFAULT_NUISANCE_PARAMS`）。
        seed: 乱数シード。

    Returns:
        未学習の `LGBMRegressor`。
    """
    merged = {**DEFAULT_NUISANCE_PARAMS, **(params or {})}
    return get_model_spec(NUISANCE_MODEL).build(Task.REGRESSION, merged, seed)


@dataclass
class NuisanceTuningResult:
    """nuisance モデルのチューニング結果。

    Attributes:
        target: 対象（"Y" または "T" など、表示用の名前）。
        best_params: 最良試行のパラメータ（探索しない固定のパラメータとの合成済み）。
        best_value: 最良試行の out-of-fold MSE（fold 平均）。
        baseline_value: 既定パラメータでの out-of-fold MSE（改善幅の比較用）。
        trials: 試行一覧（番号・状態・MSE・パラメータ）。
        study: Optuna の study（最適化履歴の図に使う）。
        used_baseline: 既定パラメータの方が良かったので、`best_params` を既定にしたか。
    """

    target: str
    best_params: dict[str, Any]
    best_value: float
    baseline_value: float
    trials: pd.DataFrame
    study: optuna.Study = field(repr=False)
    used_baseline: bool = False

    def rescaled_mse(self, factor: float) -> NuisanceTuningResult:
        """MSE（試行ごとの値・最良値・既定の値）を `factor` 倍した新しい結果を返す。

        目的変数を定数 c 倍してチューニングしたとき、`factor=1/c²` で元の単位の MSE に戻す。
        パラメータ・study はそのまま（study の値は c 倍した単位のまま）。

        Args:
            factor: MSE に掛ける倍率（正の値）。

        Returns:
            MSE を掛けた結果。

        Raises:
            ValueError: `factor` が正の有限な値でない場合。
        """
        if not (np.isfinite(factor) and factor > 0):
            raise ValueError(f"factor は正の有限な値にしてください: {factor}")
        return replace(
            self,
            best_value=self.best_value * factor,
            baseline_value=self.baseline_value * factor,
            trials=self.trials.assign(value=self.trials["value"] * factor),
        )


def cross_validated_mse(
    model: BaseEstimator,
    features: np.ndarray,
    target: np.ndarray,
    cv: Splitter,
    trial: optuna.Trial | None = None,
) -> float:
    """cross-fitting と同じ分割での out-of-fold MSE（fold 平均）を返す。

    Args:
        model: 未学習のモデル（fold ごとに clone して使う）。
        features: 特徴量（`[X, W]`）。
        target: 目的変数（Y または T）。
        cv: 分割器。
        trial: Optuna の試行（渡すと fold ごとに途中経過を報告し、枝刈りを判定する）。

    Returns:
        fold ごとの MSE の平均。

    Raises:
        optuna.TrialPruned: 途中経過が悪く、枝刈りされた場合。
    """
    scores: list[float] = []
    for step, (train_idx, valid_idx) in enumerate(cv.split(features)):
        fitted = clone(model).fit(features[train_idx], target[train_idx])
        residual = target[valid_idx] - fitted.predict(features[valid_idx])
        scores.append(float(np.mean(residual**2)))
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def cross_fit_predict(
    model: BaseEstimator,
    features: np.ndarray,
    target: np.ndarray,
    cv: Splitter,
) -> np.ndarray:
    """cross-fitting の out-of-fold の予測を返す（各行は、その行を学習に使っていないモデルで予測）。

    Args:
        model: 未学習のモデル（fold ごとに clone して使う）。
        features: 特徴量。
        target: 目的変数。
        cv: 分割器（検証側の行が全行をちょうど1回ずつ覆うもの。K-fold・`BlockedTimeSeriesSplit`）。

    Returns:
        行ごとの予測（`target` と同じ長さ）。

    Raises:
        ValueError: 検証側に一度も入らない行がある場合。
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    prediction = np.full(len(target), np.nan)
    for train_idx, valid_idx in cv.split(features):
        fitted = clone(model).fit(features[train_idx], target[train_idx])
        prediction[valid_idx] = fitted.predict(features[valid_idx])
    if np.isnan(prediction).any():
        raise ValueError("検証側に一度も入らない行があります（分割器が全行を覆っていません）")
    return prediction


def _suggest(trial: optuna.Trial, search_space: SearchSpace | None) -> dict[str, Any]:
    """探索空間から1試行分のパラメータをサンプリングする。"""
    if search_space is None:
        # 既存の LightGBM の探索空間（modeling.models.gbdt）をそのまま使う
        return get_model_spec(NUISANCE_MODEL).search_space(trial, Task.REGRESSION)
    if callable(search_space):
        return search_space(trial)
    # YAML と同じ書式の dict（{名前: {type, low, high, ...}}）。mlflow などの import を
    # 必要なときだけにするため、ここで読み込む
    from modeling.tuning import suggest_from_space

    return suggest_from_space(trial, search_space)


def tune_nuisance(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    cv: Splitter,
    n_trials: int,
    seed: int = 0,
    target_name: str = "",
    search_space: SearchSpace | None = None,
    timeout: float | None = None,
    base_params: dict[str, Any] | None = None,
) -> NuisanceTuningResult:
    """nuisance モデルのハイパーパラメータを Optuna で探索する。

    Args:
        features: 特徴量（`[X, W]` の順に横に並べたもの）。
        target: 目的変数（Y または T）。
        cv: 分割器（cross-fitting と同じ分割方法・分割数のもの。`make_cv` で作る）。
        n_trials: 試行回数（1以上）。
        seed: 乱数シード（サンプラーとモデル）。
        target_name: 結果の表示用の名前。
        search_space: 探索空間。None なら LightGBM の既定の探索空間。dict（YAML の書式）か、
            `trial` を受け取ってパラメータの dict を返す関数。
        timeout: 探索の打ち切り時間（秒）。
        base_params: 探索しないパラメータ（例: `n_jobs`）。None なら `DEFAULT_NUISANCE_PARAMS`。
            比較の基準（`baseline_value`）にもこれを使う。

    Returns:
        チューニング結果。最良パラメータは `base_params` に探索結果を上書きした完全な形。
        探索したどの試行も既定パラメータ（`base_params`）より悪ければ、既定パラメータを返す。

    Raises:
        ValueError: `n_trials` が1未満、または入力に欠損・inf がある場合。
    """
    if n_trials < 1:
        raise ValueError(f"n_trials は1以上にしてください: {n_trials}")
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).ravel()
    check_finite("特徴量", x)
    check_finite(target_name or "目的変数", y)

    base = {**DEFAULT_NUISANCE_PARAMS, **(base_params or {})}
    baseline = cross_validated_mse(make_nuisance_model(base, seed), x, y, cv)

    def objective(trial: optuna.Trial) -> float:
        params = {**base, **_suggest(trial, search_space)}
        # 探索空間が返す固定値（suggest しない値）も含めて、試行のパラメータをそのまま残す
        trial.set_user_attr("params", params)
        return cross_validated_mse(make_nuisance_model(params, seed), x, y, cv, trial)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        study_name=f"nuisance_{target_name}",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    trials = study.trials_dataframe(attrs=("number", "state", "value", "params"))
    trials.insert(0, "target", target_name)
    # 探索した中に既定パラメータより良いものが無ければ（試行が少ないときなど）、既定を使う
    use_baseline = baseline <= study.best_value
    return NuisanceTuningResult(
        target=target_name,
        best_params=dict(base) if use_baseline else dict(study.best_trial.user_attrs["params"]),
        best_value=baseline if use_baseline else float(study.best_value),
        baseline_value=baseline,
        trials=trials,
        study=study,
        used_baseline=use_baseline,
    )
