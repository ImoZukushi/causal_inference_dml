"""CATE の推定結果と感度分析の図。

どの関数も `Figure` を返すだけで、保存・表示はしない
（ノートブックで表示し、`output.OutputSaver` で保存する）。
欠損・inf の点は描かずに除く。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.container import BarContainer
from matplotlib.patches import Patch

from causal_inference.estimators import ci_columns, ci_level_label
from causal_inference.nuisance import NuisanceTuningResult
from util.plotting import ensure_japanese_font

TRUE_COLOR = "black"


def setup_style() -> None:
    """図のテーマと日本語フォントを設定する（図を作る前に1回呼ぶ）。"""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    ensure_japanese_font()


def _effect_text(treatment_change: str | None) -> str:
    """効果の説明（例: "T: 0 → 1 のときの Y の変化"）。指定が無ければ T を1単位増やしたとき。"""
    if treatment_change:
        return f"{treatment_change} のときの Y の変化"
    return "T を1単位増やしたときの Y の変化"


def _with_change(text: str, treatment_change: str | None) -> str:
    """図のタイトルに処置の変化（基準値 → 目標値）を添える。"""
    return f"{text}（{treatment_change}）" if treatment_change else text


def _finite(*arrays: np.ndarray) -> np.ndarray:
    """すべての配列で有限な位置（True）を返す。"""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(np.asarray(array, dtype=float))
    return mask


def plot_cv_folds(splitter: Any, n_rows: int, title: str = "cross-fitting の分割") -> plt.Figure:
    """分割ごとに、各行が学習・検証・除外（gap）のどれに使われるかを帯で示す。

    Args:
        splitter: `split` を持つ分割器（`nuisance.make_cv` の結果）。
        n_rows: 行数（時系列なら時間順の位置）。
        title: 図のタイトル。

    Returns:
        図。
    """
    folds = list(splitter.split(np.zeros((n_rows, 1))))
    # 0: 除外（gap）、1: 学習、2: 検証
    usage = np.zeros((len(folds), n_rows))
    for i, (train, test) in enumerate(folds):
        usage[i, train] = 1
        usage[i, test] = 2
    colors = ["#d9d9d9", *sns.color_palette("muted", 2).as_hex()]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(folds) + 1.6), constrained_layout=True)
    ax.imshow(usage, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(colors), vmin=0, vmax=2)  # fmt: skip
    ax.set_yticks(np.arange(len(folds)), [f"fold {i + 1}" for i in range(len(folds))])
    ax.set_xlabel("行の位置（時系列なら時間順）")
    ax.grid(False)
    handles = [
        Patch(color=c, label=label)
        for c, label in zip(colors, ["除外（gap）", "学習", "検証"], strict=True)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title(f"{title}（{splitter!r}、{n_rows:,} 行）")
    return fig


def plot_ate_comparison(
    ate_table: pd.DataFrame,
    true_ate: float | None = None,
    title: str = "ATE の推定値の比較",
    treatment_change: str | None = None,
) -> plt.Figure:
    """推定器ごとの ATE の点推定と CI を横に並べて比べる。

    Args:
        ate_table: 列 `estimator`, `ate`, `ate_lower`, `ate_upper` の表
            （ベースラインを含めてよい）。
        true_ate: 真の ATE（合成データのとき。縦線で示す）。
        title: 図のタイトル。
        treatment_change: 処置の変化の表示（例: "T: 0 → 1"。`estimators.treatment_change_label`）。
            指定すると横軸の説明に使う。

    Returns:
        図。
    """
    table = ate_table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(table) + 1.8), constrained_layout=True)
    positions = np.arange(len(table))[::-1]
    errors = np.vstack([table["ate"] - table["ate_lower"], table["ate_upper"] - table["ate"]])
    colors = sns.color_palette("muted", len(table))
    for pos, (_, row), err, color in zip(
        positions, table.iterrows(), errors.T, colors, strict=True
    ):
        ax.errorbar(row["ate"], pos, xerr=err.reshape(2, 1), fmt="o", capsize=5, color=color)
        ax.annotate(f"{row['ate']:.4g}", (row["ate"], pos), xytext=(0, 8),
                    textcoords="offset points", ha="center")  # fmt: skip
    ax.axvline(0, color="gray", lw=1, ls=":")
    if true_ate is not None:
        ax.axvline(true_ate, color=TRUE_COLOR, lw=1.5, ls="--", label=f"真の ATE = {true_ate:.4g}")
        ax.legend(loc="lower right")
    ax.set_yticks(positions, table["estimator"])
    ax.set_xlabel(f"ATE（{_effect_text(treatment_change)}）")
    ci_level = table["ci_level"].iloc[0] if "ci_level" in table else 0.95
    ax.set_title(f"{_with_change(title, treatment_change)}\n点: 推定値、線: {ci_level:.0%} CI")
    return fig


def plot_cate_true_vs_estimated(
    true_cate: np.ndarray,
    estimates: Mapping[str, pd.DataFrame],
    max_points: int = 3000,
    seed: int = 0,
    treatment_change: str | None = None,
) -> plt.Figure:
    """真の CATE と推定した CATE の散布図（推定器ごと）。対角線に乗るほど良い。

    Args:
        true_cate: 行ごとの真の CATE（`estimates` と同じ処置の変化での値）。
        estimates: {推定器名: 列 `cate` を持つ表}（行の並びは `true_cate` と同じ）。
        max_points: 描く点の上限（多いときは無作為に間引く）。
        seed: 間引きの乱数シード。
        treatment_change: 処置の変化の表示（例: "T: 0 → 1"）。タイトルに添える。

    Returns:
        図。
    """
    truth = np.asarray(true_cate, dtype=float)
    fig, axes = plt.subplots(
        1,
        len(estimates),
        figsize=(5.5 * len(estimates), 5.2),
        squeeze=False,
        constrained_layout=True,
    )
    rng = np.random.default_rng(seed)
    for ax, (name, frame) in zip(axes[0], estimates.items(), strict=True):
        estimate = frame["cate"].to_numpy()
        valid = np.flatnonzero(_finite(truth, estimate))
        if len(valid) > max_points:
            valid = rng.choice(valid, max_points, replace=False)
        ax.scatter(truth[valid], estimate[valid], s=6, alpha=0.35)
        low = min(truth[valid].min(), estimate[valid].min())
        high = max(truth[valid].max(), estimate[valid].max())
        ax.plot([low, high], [low, high], color=TRUE_COLOR, lw=1, ls="--", label="推定 = 真値")
        corr = np.corrcoef(truth[valid], estimate[valid])[0, 1]
        rmse = np.sqrt(np.mean((truth[valid] - estimate[valid]) ** 2))
        ax.set_title(f"{name}\n相関 {corr:.3f}、RMSE {rmse:.4g}")
        ax.set_xlabel("真の CATE")
        ax.set_ylabel("推定した CATE")
        ax.legend(loc="upper left")
    fig.suptitle(
        _with_change(f"真の CATE と推定した CATE（{len(valid):,} 点を表示）", treatment_change)
    )
    return fig


def plot_cate_distribution(
    estimates: Mapping[str, pd.DataFrame],
    true_cate: np.ndarray | None = None,
    treatment_change: str | None = None,
) -> plt.Figure:
    """推定した CATE の分布（真の CATE があれば重ねる）。

    Args:
        estimates: {推定器名: 列 `cate` を持つ表}。
        true_cate: 行ごとの真の CATE。
        treatment_change: 処置の変化の表示（例: "T: 0 → 1"）。横軸・タイトルに使う。

    Returns:
        図。
    """
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, frame in estimates.items():
        values = frame["cate"].to_numpy()
        sns.kdeplot(values[np.isfinite(values)], ax=ax, label=name, fill=True, alpha=0.2)
    if true_cate is not None:
        truth = np.asarray(true_cate, dtype=float)
        sns.kdeplot(truth[np.isfinite(truth)], ax=ax, label="真の CATE", color=TRUE_COLOR, ls="--")
    ax.set_xlabel(f"CATE（{_effect_text(treatment_change)}）")
    ax.set_ylabel("密度")
    ax.set_title(_with_change("CATE の分布（効果の異質性）", treatment_change))
    ax.legend()
    return fig


def plot_cate_by_feature(
    x: pd.DataFrame,
    estimates: Mapping[str, pd.DataFrame],
    true_cate: np.ndarray | None = None,
    features: Sequence[str] | None = None,
    n_bins: int = 10,
    n_cols: int = 2,
    ci_alphas: Sequence[float] = (0.05, 0.01),
    band_alphas: Sequence[float] = (0.35, 0.15),
    treatment_change: str | None = None,
) -> plt.Figure:
    """X の各変数の分位ごとの CATE の平均（どの変数で効果が変わるか）。

    変数を分位で `n_bins` 個に分け、各区間の CATE の点推定の平均を線で、各信頼水準の CI の
    下限・上限の平均を帯で描く。帯は狭い CI（例: 95%）ほど濃く、広い CI（例: 99%）ほど薄く塗る。
    他の変数の影響も混ざった周辺的な関係である点に注意。

    Args:
        x: X（`estimates` と同じ行の並び）。
        estimates: {推定器名: 列 `cate` と、`ci_alphas` の各水準の CI の列
            （`estimators.ci_columns` の列名。例: `cate_lower_95`）を持つ表}
            （`summarize_effects(..., extra_alphas=...)` の結果）。
        true_cate: 行ごとの真の CATE（あれば黒の破線で重ねる）。
        features: 描く変数（None なら X の全列）。
        n_bins: 分位の数。
        n_cols: 1行に並べる図の数。
        ci_alphas: 帯で描く CI の有意水準（例: (0.05, 0.01) で 95% と 99%）。
        band_alphas: 帯の塗りの不透明度。CI の狭い順（信頼水準の低い順）に対応させる。
        treatment_change: 処置の変化の表示（例: "T: 0 → 1"）。タイトルに使う。

    Returns:
        図。

    Raises:
        KeyError: `ci_alphas` の CI の列が `estimates` の表に無い場合。
        ValueError: `band_alphas` の数が `ci_alphas` より少ない場合。
    """
    # 狭い CI（有意水準の大きい順）から並べ、濃い塗りを対応させる
    levels = sorted(dict.fromkeys(ci_alphas), reverse=True)
    if len(band_alphas) < len(levels):
        raise ValueError(f"band_alphas は {len(levels)} 個以上必要です: {band_alphas}")
    for name, frame in estimates.items():
        missing = [c for a in levels for c in ci_columns(a) if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} の表に CI の列がありません: {missing}")

    features = list(features) if features is not None else list(x.columns)
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5 * n_cols, 3.2 * n_rows), sharey=True, squeeze=False,
        constrained_layout=True,
    )  # fmt: skip
    colors = dict(zip(estimates, sns.color_palette("muted", len(estimates)), strict=True))
    # 推定器が1つなら凡例に推定器名を付けない
    prefix = {name: (f"{name} " if len(estimates) > 1 else "") for name in estimates}
    for ax, feature in zip(axes.flat, features, strict=False):
        values = x[feature].to_numpy(dtype=float)
        # 同じ値が多い変数でも区間を作れるよう、重複した境界はまとめる
        bins = pd.qcut(pd.Series(values), q=n_bins, duplicates="drop")
        centers = pd.Series(values).groupby(bins, observed=True).mean()
        for name, frame in estimates.items():
            means = frame.reset_index(drop=True).groupby(bins, observed=True).mean()
            # 広い CI（薄い帯）を先に描き、その上に狭い CI（濃い帯）と点推定を重ねる
            for level, band_alpha in reversed(list(zip(levels, band_alphas, strict=False))):
                lower_col, upper_col = ci_columns(level)
                ax.fill_between(
                    centers, means[lower_col], means[upper_col], color=colors[name],
                    alpha=band_alpha, lw=0, label=f"{prefix[name]}{ci_level_label(level)}% CI",
                )  # fmt: skip
            ax.plot(centers, means["cate"], marker="o", ms=3, color=colors[name],
                    label=f"{prefix[name]}点推定")  # fmt: skip
        if true_cate is not None:
            truth = pd.Series(np.asarray(true_cate, dtype=float))
            ax.plot(centers, truth.groupby(bins, observed=True).mean(), color=TRUE_COLOR,
                    ls="--", lw=1.2, label="真の CATE")  # fmt: skip
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(feature)
        ax.set_xlabel(f"{feature}（区間の平均）")
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("CATE の平均")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    level_text = "・".join(f"{ci_level_label(a)}%" for a in levels)
    fig.suptitle(
        f"変数の分位（{n_bins}区間）ごとの CATE の平均（{_effect_text(treatment_change)}）\n"
        f"（線: 点推定、帯: {level_text} CI の下限・上限の平均。濃い帯ほど信頼水準が低い）"
    )
    return fig


def plot_feature_importance(importances: pd.Series, title: str) -> plt.Figure:
    """変数の重要度の横棒グラフ（大きい順）。

    Args:
        importances: {変数名: 重要度} の Series。
        title: 図のタイトル。

    Returns:
        図。
    """
    ordered = importances.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(ordered) + 1.5), constrained_layout=True)
    ax.barh(ordered.index, ordered.to_numpy())
    ax.set_xlim(left=0)
    ax.set_xlabel("重要度")
    ax.set_title(title)
    return fig


def plot_linear_coefficients(coefficients: pd.DataFrame, title: str) -> plt.Figure:
    """LinearDML の θ(X) の係数と CI（切片を除く）。

    Args:
        coefficients: 列 `term`, `coef`, `lower`, `upper` の表（`linear_dml_coefficients` の結果）。
        title: 図のタイトル。

    Returns:
        図。
    """
    table = coefficients[coefficients["term"] != "intercept"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(table) + 1.5), constrained_layout=True)
    significant = (table["lower"] > 0) | (table["upper"] < 0)
    palette = sns.color_palette("muted")
    colors = [palette[3] if flag else palette[0] for flag in significant]
    errors = np.vstack([table["coef"] - table["lower"], table["upper"] - table["coef"]])
    for pos, (coef, err, color) in enumerate(zip(table["coef"], errors.T, colors, strict=True)):
        ax.errorbar(coef, pos, xerr=err.reshape(2, 1), fmt="o", capsize=3, color=color)
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.set_yticks(np.arange(len(table)), table["term"])
    ax.set_xlabel("係数（その変数が1増えたときの CATE の変化）")
    ax.set_title(f"{title}（赤: CI が 0 を含まない）")
    return fig


def plot_tuning_history(results: Sequence[NuisanceTuningResult]) -> plt.Figure:
    """Optuna の試行ごとの out-of-fold MSE と、それまでの最良値（nuisance モデルごと）。

    Args:
        results: nuisance モデルのチューニング結果。

    Returns:
        図。
    """
    fig, axes = plt.subplots(
        1, len(results), figsize=(6 * len(results), 4.5), squeeze=False, constrained_layout=True
    )
    for ax, result in zip(axes[0], results, strict=True):
        trials = result.trials[result.trials["state"] == "COMPLETE"]
        ax.scatter(trials["number"], trials["value"], s=14, alpha=0.6, label="各試行")
        ax.plot(trials["number"], trials["value"].cummin(), color="C1", label="それまでの最良")
        ax.axhline(result.baseline_value, color="gray", ls="--", label="既定パラメータ")
        # 極端に悪い試行で縦軸が潰れないよう、既定パラメータの2倍までを表示する
        if len(trials) and trials["value"].max() > 2 * result.baseline_value:
            ax.set_ylim(top=2 * result.baseline_value)
        ax.set_xlabel("試行番号")
        ax.set_ylabel("out-of-fold MSE")
        improvement = 1 - result.best_value / result.baseline_value
        suffix = "、既定を採用" if result.used_baseline else ""
        ax.set_title(f"{result.target} のモデル（既定比 {improvement:+.1%} 改善{suffix}）")
        ax.legend()
    fig.suptitle("nuisance モデルの Optuna による探索")
    return fig


def plot_ovb_contour(
    grid: pd.DataFrame,
    theta: float,
    robustness_value: float | None = None,
    null_hypothesis: float = 0.0,
    bound: str = "ci",
    treatment_change: str | None = None,
) -> plt.Figure:
    """未観測交絡の強さ（c_t, c_y）ごとの、ATE の 0 に近い側の限界の等高線図。

    効果が正なら下限、負なら上限を描く。太線（null_hypothesis の等高線）より右上の交絡があると、
    効果が無いことを否定できなくなる。

    Args:
        grid: `sensitivity.ovb_grid` の結果。
        theta: ATE の点推定（効果の向きを決める）。
        robustness_value: robustness value（対角線上に印を付ける）。
        null_hypothesis: 効果が無いとみなす値。
        bound: "ci"（標本誤差も含めた範囲）か "theta"（点推定の範囲）。
        treatment_change: 処置の変化の表示（例: "T: 0 → 1"）。タイトルに添える。

    Returns:
        図。
    """
    column = f"{bound}_lower" if theta >= null_hypothesis else f"{bound}_upper"
    table = grid.pivot(index="c_y", columns="c_t", values=column)
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    contour = ax.contourf(table.columns, table.index, table.to_numpy(), levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=f"ATE の{'下限' if column.endswith('lower') else '上限'}")
    values = table.to_numpy()
    if np.nanmin(values) <= null_hypothesis <= np.nanmax(values):
        lines = ax.contour(table.columns, table.index, values, levels=[null_hypothesis],
                           colors="white", linewidths=2.5)  # fmt: skip
        ax.clabel(lines, fmt={null_hypothesis: f"ATE = {null_hypothesis:g}"})
    if robustness_value is not None and np.isfinite(robustness_value):
        ax.scatter([robustness_value], [robustness_value], color="red", zorder=3,
                   label=f"robustness value = {robustness_value:.3f}")  # fmt: skip
        ax.legend(loc="lower right")
    ax.set_xlabel("c_t（未観測交絡が処置の残差を説明する割合）")
    ax.set_ylabel("c_y（未観測交絡が結果の残差を説明する割合）")
    label = "CI" if bound == "ci" else "点推定"
    title = f"未観測交絡の強さと ATE の範囲（{label}、点推定 {theta:.4g}）"
    ax.set_title(f"{title}\n{treatment_change}" if treatment_change else title)
    return fig


def plot_e_values(e_values: pd.DataFrame) -> plt.Figure:
    """推定器・処置の変化幅ごとの E-value（点推定と CI）の棒グラフ。

    Args:
        e_values: `sensitivity.e_value_table` の結果。

    Returns:
        図。
    """
    table = e_values.melt(
        id_vars=["estimator", "delta_label"],
        value_vars=["e_value_point", "e_value_ci"],
        var_name="kind",
        value_name="e_value",
    )
    table["kind"] = table["kind"].map({"e_value_point": "点推定", "e_value_ci": "CI の限界"})
    table["group"] = table["estimator"] + "\n" + table["delta_label"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=table, x="group", y="e_value", hue="kind", ax=ax)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.2f", padding=2)
    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("")
    ax.set_ylabel("E-value（リスク比のスケール）")
    ax.set_title("E-value: 効果を打ち消すのに必要な未観測交絡の強さ")
    ax.legend(title="")
    return fig


def plot_e_value_curve(rr: float, e_value: float, label: str) -> plt.Figure:
    """効果をちょうど打ち消す（処置–交絡、交絡–結果）のリスク比の組の曲線と E-value。

    Args:
        rr: 推定した効果を換算したリスク比。
        e_value: その E-value。
        label: 対象の説明（タイトルに使う）。

    Returns:
        図。
    """
    from causal_inference.sensitivity import e_value_bias_curve

    curve = e_value_bias_curve(rr)
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    if not curve.empty:
        ax.plot(curve["rr_treatment_confounder"], curve["rr_confounder_outcome"], lw=2)
        upper = curve["rr_treatment_confounder"].max()
        ax.fill_between(curve["rr_treatment_confounder"], curve["rr_confounder_outcome"], upper,
                        alpha=0.15, label="効果を打ち消しうる領域")  # fmt: skip
        ax.scatter([e_value], [e_value], color="red", zorder=3, label=f"E-value = {e_value:.2f}")
        ax.set_xlim(1, upper)
        ax.set_ylim(1, upper)
        ax.legend(loc="upper right")
    ax.set_xlabel("処置と未観測交絡の関連（リスク比）")
    ax.set_ylabel("未観測交絡と結果の関連（リスク比）")
    ax.set_title(f"E-value の曲線（{label}、RR = {rr:.2f}）")
    return fig


def plot_refutations(refutations: pd.DataFrame, treatment_change: str | None = None) -> plt.Figure:
    """反証テストごとの、元の推定値と新しい推定値の比較。

    Args:
        refutations: `sensitivity.run_refuters` の結果。
        treatment_change: 処置の変化の表示（例: "T: 0 → 1"）。横軸・タイトルに使う。

    Returns:
        図。
    """
    table = refutations.reset_index(drop=True).iloc[::-1]
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(table) + 1.8), constrained_layout=True)
    ax.barh(positions, table["new_effect"], color="C0", alpha=0.7, label="反証テストでの推定値")
    ax.scatter(table["estimated_effect"], positions, color="C3", marker="|", s=400, zorder=3,
               label="元の推定値（偽の結果では期待値 0）")  # fmt: skip
    for pos, (_, row) in zip(positions, table.iterrows(), strict=True):
        text = row["judgement"] if not row["error"] else "失敗"
        if np.isfinite(row["p_value"]):
            text += f"（p={row['p_value']:.2f}）"
        ax.annotate(text, (0, pos), xytext=(4, 0), textcoords="offset points", va="center")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.set_yticks(positions, table["refuter"])
    ax.set_xlabel(f"推定値（ATE。{_effect_text(treatment_change)}）")
    ax.set_title(_with_change("DoWhy の反証テスト", treatment_change))
    fig.legend(loc="outside lower center", ncol=2)
    return fig
