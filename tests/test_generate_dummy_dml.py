"""scripts/generate_dummy_dml.py（DML 用の合成データの生成）のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import generate_dummy_dml as gen
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


def test_columns_and_dtypes() -> None:
    result = gen.make_dummy_dml(n=500, seed=0)
    expected = ["Y", "T", *[f"X{i}" for i in range(1, 16)], *[f"W{i}" for i in range(1, 31)]]
    assert list(result.data.columns) == expected
    assert (result.data.dtypes == "float64").all()
    assert not result.data.isna().any().any()
    assert list(result.truth.columns) == ["true_cate", "true_t_mean"]
    assert len(result.truth) == len(result.data)


def test_reproducible_with_seed() -> None:
    a = gen.make_dummy_dml(n=200, seed=1)
    b = gen.make_dummy_dml(n=200, seed=1)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert not a.data.equals(gen.make_dummy_dml(n=200, seed=2).data)


def test_sample_ate_close_to_population_ate() -> None:
    result = gen.make_dummy_dml(n=50000, seed=0)
    assert result.truth["true_cate"].mean() == pytest.approx(gen.POPULATION_ATE, abs=0.03)


def test_cate_depends_only_on_x1_to_x5() -> None:
    result = gen.make_dummy_dml(n=300, seed=0)
    x = result.data[gen.CONFOUNDER_COLS].to_numpy()
    shuffled = x.copy()
    # X6〜X15 を入れ替えても θ(X) は変わらない
    shuffled[:, 5:] = np.random.default_rng(0).permutation(shuffled[:, 5:])
    np.testing.assert_allclose(gen.true_cate(x), gen.true_cate(shuffled))
    np.testing.assert_allclose(gen.true_cate(x), result.truth["true_cate"].to_numpy())


def test_x_confounds_and_w_does_not_affect_treatment() -> None:
    data = gen.make_dummy_dml(n=20000, seed=0).data
    # X は T に効く（T を X で回帰した決定係数が十分大きい）
    r2_x = (
        LinearRegression()
        .fit(data[gen.CONFOUNDER_COLS], data["T"])
        .score(data[gen.CONFOUNDER_COLS], data["T"])
    )
    assert r2_x > 0.3
    # W は T と無関係
    r2_w = (
        LinearRegression()
        .fit(data[gen.COVARIATE_COLS], data["T"])
        .score(data[gen.COVARIATE_COLS], data["T"])
    )
    assert r2_w < 0.01
    # W は Y に効く
    assert abs(np.corrcoef(data["W1"], data["Y"])[0, 1]) > 0.05


def test_naive_estimate_is_biased_but_true_nuisances_recover_ate() -> None:
    result = gen.make_dummy_dml(n=20000, seed=0)
    data, truth = result.data, result.truth
    # 調整しない回帰係数は交絡でずれる
    naive = np.polyfit(data["T"], data["Y"], 1)[0]
    assert abs(naive - gen.POPULATION_ATE) > 0.2
    # 真の m(X) で残差化した T を使えば、(分散重み付き) 平均効果 = ATE に近い値になる
    t_res = data["T"] - truth["true_t_mean"]
    x = data[gen.CONFOUNDER_COLS].to_numpy()
    w = data[gen.COVARIATE_COLS].to_numpy()
    y_res = data["Y"] - gen.outcome_baseline(x, w) - truth["true_cate"] * truth["true_t_mean"]
    estimate = (t_res * y_res).sum() / (t_res**2).sum()
    assert estimate == pytest.approx(gen.POPULATION_ATE, abs=0.05)


def test_unobserved_confounding_shifts_treatment() -> None:
    base = gen.make_dummy_dml(n=5000, seed=0)
    confounded = gen.make_dummy_dml(n=5000, seed=0, confounding_strength=1.0)
    # X・W は同じで、T と Y だけが変わる
    pd.testing.assert_frame_equal(
        base.data[gen.CONFOUNDER_COLS + gen.COVARIATE_COLS],
        confounded.data[gen.CONFOUNDER_COLS + gen.COVARIATE_COLS],
    )
    assert not base.data["T"].equals(confounded.data["T"])


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"confounding_strength": -0.1}])
def test_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        gen.make_dummy_dml(**kwargs)


def test_main_writes_files(tmp_path: Path) -> None:
    gen.main(["--n", "100", "--output-dir", str(tmp_path)])
    data = pd.read_csv(tmp_path / gen.DATA_FILENAME)
    truth = pd.read_csv(tmp_path / gen.TRUTH_FILENAME)
    summary = json.loads((tmp_path / gen.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert data.shape == (100, 47)
    assert len(truth) == 100
    assert summary["population_ate"] == gen.POPULATION_ATE
    assert summary["sample_ate"] == pytest.approx(truth["true_cate"].mean())
