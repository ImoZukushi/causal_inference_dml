"""因果効果の推定（DML）と感度分析の動作確認用に、真の効果が分かる合成データを作る。

1行 = 1個体のクロスセクションデータで、列は次のとおり（いずれも float）。

| 列 | 役割 | DAG 上の位置 |
|---|---|---|
| `Y` | 結果変数 | |
| `T` | 処置変数（連続値） | X → T → Y |
| `X1`〜`X15` | 調整する交絡因子。CATE を評価する変数 | X → T、X → Y、X が効果の大きさを変える |
| `W1`〜`W30` | 精度向上のための共変量 | W → Y のみ（T には効かない） |

データ生成過程（部分線形モデル。DML の PLR が想定する形）:

    T = m(X) + κ·U + η,               η ~ N(0, 1)
    Y = θ(X)·T + g(X) + h(W) + κ·U + ε,  ε ~ N(0, 1)

- X: 平均0・分散1、隣り合う番号ほど相関する多変量正規分布（相関 ρ^|i-j|）。
- W: 互いに独立な N(0, 1)。X・T とも独立なので交絡因子ではない。
- m(X): 処置の割り当て。X1〜X15 のすべてが効く（強さは番号が大きいほど弱い）。非線形項を含む。
- g(X): X から Y への直接の効果。X1〜X15 のすべてが効く。非線形項を含む。
- h(W): W から Y への効果。W1〜W10 は線形、W11〜W20 は非線形、W21〜W30 は弱い線形。
- θ(X): 処置効果（CATE）。T を1単位増やしたときの Y の変化で、T について線形。
  X1〜X5 だけで変わり、X6〜X15 では変わらない（効果の異質性が無い変数の確認用）。

      θ(X) = 1 + 0.5·X1 − 0.4·X2 + 0.6·(1[X3 > 0] − 0.5) + 0.3·(X4·X5 − ρ)

  各項の母集団平均は0なので、母集団の ATE は 1.0。
- U: 観測されない交絡因子（出力しない）。強さ κ（`--confounding-strength`）の既定は0で、
  既定では「X を調整すれば識別できる」という前提がそのまま成り立つ。κ > 0 にすると
  未観測交絡で推定値がずれるデータになり、感度分析の動作確認に使える。

η を等分散にしているため、PLR の推定量が収束する「分散で重み付けした平均効果」
E[Var(T|X)·θ(X)] / E[Var(T|X)] は、ATE と一致する。

前処理の動作確認用に、値を作った後で欠損（NaN）と ±inf を入れられる（`--missing-rate`・
`--inf-rate`。既定はどちらも0）。分析用データの全列に、列ごと独立に入れる。真値には入れない。

出力（いずれも gitの管理外）:

| ファイル | 内容 |
|---|---|
| `data/synthetic/dml/dummy_dml.csv` | 分析用データ（`Y`, `T`, `X1`〜`X15`, `W1`〜`W30`） |
| `data/synthetic/dml/dummy_dml_truth.csv` | 行ごとの真値（`true_cate`, `true_t_mean`） |
| `data/synthetic/dml/dummy_dml_truth.json` | 母集団・標本の ATE と生成の設定 |

使い方:
    uv run python scripts/generate_dummy_dml.py
    uv run python scripts/generate_dummy_dml.py --n 20000 --confounding-strength 0.5
    uv run python scripts/generate_dummy_dml.py --missing-rate 0.02 --inf-rate 0.005 \
        --output-dir data/synthetic/dml_nonfinite
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util.paths import data_dir, ensure_parent_dir  # noqa: E402

N_CONFOUNDERS = 15
N_COVARIATES = 30
OUTCOME_COL = "Y"
TREATMENT_COL = "T"
CONFOUNDER_COLS = [f"X{i}" for i in range(1, N_CONFOUNDERS + 1)]
COVARIATE_COLS = [f"W{i}" for i in range(1, N_COVARIATES + 1)]
OUTPUT_DIR = data_dir() / "synthetic" / "dml"
DATA_FILENAME = "dummy_dml.csv"
TRUTH_FILENAME = "dummy_dml_truth.csv"
SUMMARY_FILENAME = "dummy_dml_truth.json"

X_CORRELATION = 0.3  # X の隣り合う番号の相関 ρ（番号が k 離れると ρ^k）
POPULATION_ATE = 1.0  # θ(X) の母集団平均（θ の定数項）
# m(X) の線形部分の係数（X1〜X15）。番号が大きいほど弱い
TREATMENT_COEFS = np.array(
    [0.8, 0.6, -0.5, 0.4, 0.3, 0.5, -0.4, 0.3, 0.2, -0.2, 0.15, 0.1, -0.1, 0.1, 0.05]
)
# g(X) の線形部分の係数（X1〜X15）。符号を m(X) とそろえ、交絡バイアスが打ち消し合わずに
# 正の向きに出るようにする（調整しない推定は ATE を過大に見積もる）
OUTCOME_COEFS = np.array(
    [1.0, 0.8, -0.6, 0.5, 0.4, 0.6, -0.5, 0.3, 0.3, -0.2, 0.2, 0.2, -0.15, 0.1, 0.1]
)


@dataclass(frozen=True)
class DummyDmlData:
    """合成データと真値。

    Attributes:
        data: 分析用データ（`Y`, `T`, `X1`〜`X15`, `W1`〜`W30`）。
        truth: 行ごとの真値（`true_cate` = θ(X)、`true_t_mean` = E[T|X]）。
            行の並びは `data` と同じ。
    """

    data: pd.DataFrame
    truth: pd.DataFrame


def _sample_confounders(n: int, rng: np.random.Generator) -> np.ndarray:
    """X1〜X15 を作る（平均0・分散1、相関 ρ^|i-j| の多変量正規分布）。"""
    idx = np.arange(N_CONFOUNDERS)
    cov = X_CORRELATION ** np.abs(idx[:, None] - idx[None, :])
    return rng.multivariate_normal(np.zeros(N_CONFOUNDERS), cov, size=n)


def true_cate(x: np.ndarray) -> np.ndarray:
    """真の CATE θ(X) を返す（X1〜X5 だけで変わる。母集団平均は `POPULATION_ATE`）。

    Args:
        x: X1〜X15 の配列（形は (n, 15)）。

    Returns:
        行ごとの θ(X)。
    """
    x1, x2, x3, x4, x5 = (x[:, j] for j in range(5))
    return (
        POPULATION_ATE
        + 0.5 * x1
        - 0.4 * x2
        # しきい値で効果が変わる項（1[X3>0] の平均 0.5 を引いて平均0にする）
        + 0.6 * ((x3 > 0).astype(float) - 0.5)
        # 交互作用の項（E[X4·X5] = ρ を引いて平均0にする）
        + 0.3 * (x4 * x5 - X_CORRELATION)
    )


def treatment_mean(x: np.ndarray) -> np.ndarray:
    """処置の割り当て m(X) = E[T|X] を返す（未観測交絡 U を除いた部分）。

    Args:
        x: X1〜X15 の配列（形は (n, 15)）。

    Returns:
        行ごとの m(X)。
    """
    # 非線形項は平均0になるようにする（X2² の平均は1）
    return x @ TREATMENT_COEFS + 0.3 * (x[:, 1] ** 2 - 1) + 0.4 * np.tanh(2 * x[:, 5])


def outcome_baseline(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """処置によらない Y の部分 g(X) + h(W) を返す。

    Args:
        x: X1〜X15 の配列（形は (n, 15)）。
        w: W1〜W30 の配列（形は (n, 30)）。

    Returns:
        行ごとの g(X) + h(W)。
    """
    g = x @ OUTCOME_COEFS + 0.5 * np.sin(np.pi * x[:, 6]) + 0.4 * x[:, 2] * x[:, 7]
    # W1〜W10: 線形（係数 0.6〜0.15 の交互の符号）
    linear_coefs = np.linspace(0.6, 0.15, 10) * np.where(np.arange(10) % 2 == 0, 1, -1)
    h = w[:, :10] @ linear_coefs
    # W11〜W20: 非線形（線形モデルでは拾いにくい形）
    nonlinear = w[:, 10:20]
    h = h + 0.4 * np.sin(2 * nonlinear[:, :5]).sum(axis=1)
    h = h + 0.2 * (nonlinear[:, 5:] ** 2 - 1).sum(axis=1)
    # W21〜W30: 弱い線形
    h = h + w[:, 20:30] @ np.full(10, 0.05)
    return g + h


def make_dummy_dml(
    n: int = 10000, seed: int = 0, confounding_strength: float = 0.0
) -> DummyDmlData:
    """DML 用の合成データと真値を作る。

    Args:
        n: 行数（個体数）。
        seed: 乱数のシード。
        confounding_strength: 未観測交絡 U が T と Y に効く強さ κ（0 なら未観測交絡なし）。

    Returns:
        分析用データと行ごとの真値。

    Raises:
        ValueError: `n` が1未満、または `confounding_strength` が負の場合。
    """
    if n < 1:
        raise ValueError(f"n は1以上にしてください: {n}")
    if confounding_strength < 0:
        raise ValueError(f"confounding_strength は0以上にしてください: {confounding_strength}")
    rng = np.random.default_rng(seed)
    x = _sample_confounders(n, rng)
    w = rng.normal(0, 1, size=(n, N_COVARIATES))
    u = rng.normal(0, 1, n)
    m = treatment_mean(x)
    t = m + confounding_strength * u + rng.normal(0, 1, n)
    theta = true_cate(x)
    y = theta * t + outcome_baseline(x, w) + confounding_strength * u + rng.normal(0, 1, n)

    data = pd.concat(
        [
            pd.DataFrame({OUTCOME_COL: y, TREATMENT_COL: t}),
            pd.DataFrame(x, columns=CONFOUNDER_COLS),
            pd.DataFrame(w, columns=COVARIATE_COLS),
        ],
        axis=1,
    ).astype("float64")
    truth = pd.DataFrame({"true_cate": theta, "true_t_mean": m})
    return DummyDmlData(data=data, truth=truth)


def add_nonfinite(
    data: pd.DataFrame, missing_rate: float = 0.0, inf_rate: float = 0.0, seed: int = 0
) -> pd.DataFrame:
    """値を作った後で、全列に欠損（NaN）と ±inf を入れる（前処理の動作確認用）。

    列ごとに独立に、`missing_rate` の割合を NaN、残りから `inf_rate` の割合を ±inf
    （符号は半々）にする。
    値の生成とは別の乱数を使うので、入れなかった値は元と同じ。

    Args:
        data: `make_dummy_dml` の分析用データ。
        missing_rate: NaN にする割合（0以上1未満）。
        inf_rate: ±inf にする割合（0以上1未満）。
        seed: 乱数のシード。

    Returns:
        NaN・inf を入れた DataFrame（入力は変更しない）。

    Raises:
        ValueError: 割合が範囲外の場合。
    """
    for name, rate in (("missing_rate", missing_rate), ("inf_rate", inf_rate)):
        if not 0 <= rate < 1:
            raise ValueError(f"{name} は0以上1未満にしてください: {rate}")
    rng = np.random.default_rng([seed, 1])
    updates = {}
    for col in data.columns:
        values = data[col].to_numpy(copy=True)
        draw = rng.random(len(values))
        values[draw < missing_rate] = np.nan
        is_inf = (draw >= missing_rate) & (draw < missing_rate + inf_rate)
        values[is_inf] = np.where(rng.random(int(is_inf.sum())) < 0.5, np.inf, -np.inf)
        updates[col] = values
    return data.assign(**updates)


def build_summary(
    result: DummyDmlData,
    seed: int,
    confounding_strength: float,
    missing_rate: float = 0.0,
    inf_rate: float = 0.0,
) -> dict:
    """真値の要約（ATE と生成の設定）を返す。

    Args:
        result: `make_dummy_dml` の結果。
        seed: 生成に使ったシード。
        confounding_strength: 生成に使った未観測交絡の強さ κ。
        missing_rate: 入れた NaN の割合。
        inf_rate: 入れた ±inf の割合。

    Returns:
        JSON に保存する辞書。
    """
    return {
        "population_ate": POPULATION_ATE,
        "sample_ate": float(result.truth["true_cate"].mean()),
        "n": len(result.data),
        "seed": seed,
        "confounding_strength": confounding_strength,
        "missing_rate": missing_rate,
        "inf_rate": inf_rate,
        "x_correlation": X_CORRELATION,
        "cate_modifiers": CONFOUNDER_COLS[:5],
        "outcome_col": OUTCOME_COL,
        "treatment_col": TREATMENT_COL,
        "confounder_cols": CONFOUNDER_COLS,
        "covariate_cols": COVARIATE_COLS,
    }


def main(argv: list[str] | None = None) -> None:
    """合成データを作り、分析用データと真値を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10000, help="行数（個体数）")
    parser.add_argument("--seed", type=int, default=0, help="乱数のシード")
    parser.add_argument(
        "--confounding-strength",
        type=float,
        default=0.0,
        help="未観測交絡 U の強さ κ（0 なら未観測交絡なし）",
    )
    parser.add_argument("--missing-rate", type=float, default=0.0, help="列ごとの NaN の割合")
    parser.add_argument("--inf-rate", type=float, default=0.0, help="列ごとの ±inf の割合")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="出力先のディレクトリ")
    args = parser.parse_args(argv)

    result = make_dummy_dml(args.n, args.seed, args.confounding_strength)
    summary = build_summary(
        result, args.seed, args.confounding_strength, args.missing_rate, args.inf_rate
    )
    data = add_nonfinite(result.data, args.missing_rate, args.inf_rate, args.seed)

    data_path = ensure_parent_dir(args.output_dir / DATA_FILENAME)
    # inf は CSV に "inf" / "-inf" と書かれ、pandas の read_csv で inf として読める
    data.to_csv(data_path, index=False)
    truth_path = ensure_parent_dir(args.output_dir / TRUTH_FILENAME)
    result.truth.to_csv(truth_path, index=False)
    summary_path = ensure_parent_dir(args.output_dir / SUMMARY_FILENAME)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"保存: {data_path}（{result.data.shape[0]}行 × {result.data.shape[1]}列）")
    print(f"保存: {truth_path}")
    print(f"保存: {summary_path}")
    print(
        f"真の ATE: 母集団 {summary['population_ate']:.3f} / 標本 {summary['sample_ate']:.3f}、"
        f"CATE の標準偏差 {result.truth['true_cate'].std():.3f}"
    )


if __name__ == "__main__":
    main()
