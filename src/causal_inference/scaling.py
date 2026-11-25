"""効果の単位の変換（Y の定数倍の戻し方と、処置の変化幅 T0→T1 への換算）。

処置の変化幅: 部分線形モデルでは、処置を T0 から T1 に変えたときの効果は「T を1単位増やしたときの
効果」の (T1 − T0) 倍になる。econml の感度分析や OLS のように単位あたりの効果しか返さない結果は、
`rescale_columns(..., factor=T1 − T0)` で換算する（T1 < T0 なら CI の下限・上限を入れ替える）。

Y の定数倍: econml の因果フォレストの CI は、効果がとても小さい単位（分散がおよそ 1e-10 以下）に
なると、
内部の固定のしきい値のせいで標準誤差が NaN になり、CI の幅が 0 になる。Y を定数 c 倍して推定すれば
避けられる。
部分線形モデルでは Y を c 倍すると効果 θ(X)・ATE・CI もちょうど c 倍になるので、c で割れば元の単位に
戻る（統計的な意味は変わらない）。

元の単位に戻すときの扱い:

| 量 | 戻し方 |
|---|---|
| 効果（ATE・CATE・CI・OLS の T の係数・感度分析の ATE の範囲・反証テストの推定値） | 1/c 倍 |
| Y のモデルの MSE | 1/c² 倍 |
| 決定係数・変数の重要度・robustness value・E-value・p 値 | そのまま（単位によらない） |
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# 効果の単位を持つ列（元の単位に戻すときに 1/c 倍する）
EFFECT_COLUMNS = ("ate", "ate_lower", "ate_upper", "cate_mean", "cate_std")
OVB_EFFECT_COLUMNS = ("theta", "theta_lower", "theta_upper", "ci_lower", "ci_upper")
REFUTATION_EFFECT_COLUMNS = ("estimated_effect", "new_effect")
OVB_GRID_EFFECT_COLUMNS = ("theta_lower", "theta_upper", "ci_lower", "ci_upper")
# (下限, 上限) の列の組（負の倍率を掛けたときに入れ替える）
INTERVAL_PAIRS = (
    ("ate_lower", "ate_upper"),
    ("theta_lower", "theta_upper"),
    ("ci_lower", "ci_upper"),
)


def check_scale(scale: float) -> float:
    """Y に掛ける定数が使える値か確かめて、float で返す。

    Args:
        scale: Y に掛ける定数。

    Returns:
        `scale`（float）。

    Raises:
        ValueError: 正の有限な値でない場合（負の値は効果の符号が反転して紛らわしいので
            受け付けない）。
    """
    value = float(scale)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"Y に掛ける定数は正の有限な値にしてください: {scale}")
    return value


def rescale_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    factor: float,
    intervals: Sequence[tuple[str, str]] = INTERVAL_PAIRS,
) -> pd.DataFrame:
    """指定した列（あるものだけ）を `factor` 倍した新しい表を返す（入力は変更しない）。

    `factor` が負のとき（処置の目標値が基準値より小さいときなど）は、`intervals` の (下限, 上限) の
    列の組を入れ替えて、下限 ≤ 上限を保つ。

    Args:
        frame: 表。
        columns: 倍率を掛ける列の候補（表に無い列は無視する）。
        factor: 倍率（元の単位に戻すなら 1/c、単位あたりの効果を T0→T1 の効果にするなら T1−T0）。
        intervals: (下限, 上限) の列の組（両方が表にある組だけ扱う）。

    Returns:
        列を掛けた表。
    """
    present = [c for c in columns if c in frame.columns]
    scaled = frame.assign(**{c: frame[c] * factor for c in present})
    if factor < 0:
        for lower, upper in intervals:
            if lower in present and upper in present:
                scaled[[lower, upper]] = scaled[[upper, lower]].to_numpy()
    return scaled
