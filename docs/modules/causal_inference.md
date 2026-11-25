# `causal_inference` — DML による CATE の推定と感度分析

DAG の作成と estimand の同定が済んでいる前提で、econml の Double Machine Learning（DML）で CATE を推定し、
未観測交絡に対する頑健さを感度分析で確かめるための部品です。使い方の例は
`notebook/001_cate_estimation_dml.ipynb` を参照してください。

データの読み込みと前処理（inf・欠損の処理、Y/T/X/W への分割）は呼び出し側（ノートブック）で行います。
このパッケージの関数は、欠損・inf を含む入力を受け取ると `ValueError` で止まります。

## 変数の役割

| 引数 | 意味 | econml での扱い |
|---|---|---|
| `y` | 結果変数 | `Y` |
| `t` | 処置変数（連続値） | `T`（`discrete_treatment=False`） |
| `x` | 交絡因子かつ効果修飾因子 | `X`（nuisance と CATE の両方に使う） |
| `w` | 精度向上のための共変量 | `W`（nuisance にだけ使う） |

## cross-fitting の分割

`make_cv(n_splits, seed, strategy=..., gap=...)`・`build_estimator(..., cv_strategy=..., gap=...)` で選びます。

| `strategy` | 分割 | 使う場面 |
|---|---|---|
| `kfold`（既定） | 行をシャッフルした K-fold。`mc_iters` で分割を変えて繰り返せる | 行が互いに独立 |
| `timeseries` | `BlockedTimeSeriesSplit`: 時間順の連続ブロックを順に検証に使い、検証ブロックの前後 `gap` 行を学習から除く（neighbors-left-out） | 時系列（行が時間順に並んでいる。自己相関がある） |

- `timeseries` は学習に検証ブロックより未来の行も使います。DML の cross-fitting は全行の out-of-fold の
  nuisance の予測が必要で、過去だけで学習する前向きの分割（`TimeSeriesSplit`）では最初のブロックの予測が
  作れない（econml は NaN にする）ためです。効果の推定であって将来の予測ではないので、未来の行を使うこと
  自体は問題になりません。自己相関による情報の漏れは `gap` で抑えます（自己相関が続く行数が目安）。
- `timeseries` の分割は乱数を使わないので、`mc_iters > 1` はエラーにします。
- Optuna のチューニングにも同じ分割器を渡します（`tune_nuisance(cv=make_cv(...))`）。
- 行の並べ替えはしません。呼び出し側で時間順に並べてから渡します（ノートブックでは `TIME_COL`）。
- econml の CI は残差が互いに独立だと仮定しているので、残差に自己相関が残ると CI が狭くなりすぎます。

## モジュール

| モジュール | 主な関数 | 内容 |
|---|---|---|
| `nuisance` | `make_cv`, `BlockedTimeSeriesSplit`, `make_nuisance_model`, `tune_nuisance`, `cross_fit_predict`, `check_finite` | cross-fitting の分割（K-fold または時系列。`n_splits >= 2` を強制）、LightGBM の nuisance モデル、Optuna による探索（目的関数は K-fold の out-of-fold MSE。既定より良い試行が無ければ既定を返す） |
| `estimators` | `build_estimator`, `fit_estimator`, `summarize_effects`, `ci_columns`, `tune_causal_forest`, `linear_dml_coefficients`, `nuisance_scores` | `LinearDML`・`CausalForestDML` の生成（`cv_strategy`・`n_splits`・`gap`・`mc_iters` で cross-fitting）、学習、ATE・CATE の点推定と CI（`extra_alphas` で複数の信頼水準。列名は `cate_lower_95`・`cate_lower_99` など） |
| `evaluation` | `baseline_estimates`, `compare_with_truth` | OLS のベースライン、真の CATE との比較（合成データ専用） |
| `sensitivity` | `e_value_continuous`, `e_value_table`, `ovb_summary`, `ovb_grid`, `fit_dowhy_wrapper`, `default_refuters`, `run_refuters` | E-value、econml の omitted variable bias 感度分析（Chernozhukov et al. 2022）、DoWhy の反証テスト |
| `plots` | `plot_*` | 図（`Figure` を返すだけで、保存・表示はしない）。`plot_cv_folds` で分割の形を確認できる。`plot_cate_by_feature` は点推定と複数の信頼水準の CI（既定 95%・99%。狭い CI ほど濃い帯）を、既定で1行に2つずつ描く |
| `partial_r2` | `residualize`, `fit_residual_regression`, `sensitivity_statistics`, `robustness_value`, `benchmark_partial_r2`, `benchmark_table`, `partial_r2_grid`, `linear_partial_r2` | 部分決定係数による感度分析（Cinelli & Hazlett 2020。R の sensemakr と同じ式）。残差回帰の推定値の RV_q・RV_q,α、観測された共変量による benchmark、調整後の推定値・CI の格子 |
| `scaling` | `check_scale`, `rescale_columns` | Y を定数倍して推定したときに、結果を元の単位に戻す |
| `output` | `OutputSaver` | 図を `outputs/figures/<分析名>/<実行ID>/`、表を `outputs/tables/<分析名>/<実行ID>/` に保存 |

## 感度分析の使い分け

| 方法 | 何が分かるか | 実装 |
|---|---|---|
| E-value | 効果を打ち消すのに必要な未観測交絡の強さ（リスク比）。連続の Y は RR ≈ exp(0.91·効果/SD(Y)) で近似 | 自前（DoWhy の E-value は線形回帰・GLM の推定器専用のため） |
| omitted variable bias | 未観測交絡が Y・T の残差の分散を c_y・c_t だけ説明するときの ATE の範囲と robustness value | econml の `sensitivity_interval`・`robustness_value` |
| 部分決定係数 | 未観測交絡が処置・結果の残差を説明する割合（R²_{T~Z\|X}・R²_{Y~Z\|T,X}）ごとの調整後の推定値・CI、RV_q・RV_q,α、「観測された共変量 X_j の k 倍の強さ」の交絡での推定値（benchmark） | 自前（`partial_r2`。sensemakr の式を DML の残差回帰に適用） |
| 反証テスト | 偽の処置・偽の結果・無関係な共変量・部分標本・模擬的な未観測交絡で推定し直したときの振る舞い | DoWhy（econml の `est.dowhy` 経由） |

DoWhy の `add_unobserved_common_cause` の `non-parametric-partial-R2` は、econml の推定器と組み合わせると
特徴量の数が合わずに失敗する（DoWhy 0.12）ため使っていません。同じ考え方の分析は econml の
omitted variable bias 感度分析で行います。

## 部分決定係数による感度分析（`partial_r2`）

DML の残差どうしの回帰 Ỹ = θ·T̃ + e（Ỹ = Y − E[Y|X,W]、T̃ = T − E[T|X,W]。cross-fitting の残差）は、
観測された共変量を取り除いた後の単回帰です。そのため、OLS 向けの Cinelli & Hazlett（2020）の式を
そのまま使えます（部分線形モデルで同じ偏りの式が成り立つことは Chernozhukov et al. 2022 が示しています）。

```python
from causal_inference.partial_r2 import (
    benchmark_partial_r2, benchmark_table, fit_residual_regression, partial_r2_grid,
    residualize, sensitivity_statistics,
)

residuals = residualize(y, t, controls, model_y=model_y, model_t=model_t, cv=cv)
fit = fit_residual_regression(residuals)              # θ̂・標準誤差・自由度
stats = sensitivity_statistics(fit, q=1.0, alpha=0.05) # R²_{Y~T|X}・RV_q・RV_q,α
bench = benchmark_partial_r2(y, t, controls, {"X1": ["X1"]}, model_y=model_y,
                             model_t=model_t, cv=cv, full_residuals=residuals)
table = benchmark_table(fit, bench, multipliers=(1, 2, 3))   # k×X1 の交絡で調整した推定値・CI
grid = partial_r2_grid(fit, np.linspace(0, 0.4, 41))         # plots.plot_partial_r2_contours に渡す
```

- benchmark の部分決定係数は、X_j を除いて nuisance を cross-fitting し直したときの残差の分散の増え方から
  求めます（R²_{T~X_j|X_−j} = 1 − Var(T̃)/Var(T̃_−j)、R²_{Y~X_j|T,X_−j} も残差回帰の残差で同様）。
  benchmark の式は線形モデルで導かれたもので、非線形の nuisance では目安です。
- 対象は残差回帰の推定値（処置の残差の分散で重み付けした平均的な効果）です。因果フォレストの ATE
  （CATE の平均）とは少し違います。econml の `ovb_summary` の c_y・c_t も部分決定係数で、こちらは
  因果フォレストの ATE が対象です。
- 自由度は n − 1、標準誤差は古典的な OLS の式です（sensemakr の式と整合させるため）。
- 単位: `ResidualRegression.rescaled(factor)` で効果の単位を換算します（Y の定数倍を戻す `1/c`、
  T0→T1 の効果にする `T1 − T0`）。部分決定係数・RV は単位によらず、変わりません。
- 式は R の sensemakr の Darfur の例と照合しています（`tests/test_partial_r2.py`）。

## 処置の基準値・目標値（T0 → T1）

効果は「処置を基準値 T0 から目標値 T1 に変えたときの Y の変化」として求めます（既定は 0 → 1、
つまり T を1単位増やしたとき）。`summarize_effects(..., t0=, t1=)`・`baseline_estimates(..., t0=, t1=)`・
`fit_dowhy_wrapper(..., t0=, t1=)` で指定し、図には `treatment_change=treatment_change_label(t0, t1)`
を渡すとタイトル・軸に「T: T0 → T1」が表示されます。

- 部分線形モデルでは効果は θ(X)·(T1 − T0) で、差 T1 − T0 だけで決まります（基準値の水準によらない）。
- econml の感度分析（`ovb_summary`・`ovb_grid`）は1単位あたりの値を返すので、
  `rescale_columns(..., factor=T1 − T0)` で換算します。T1 < T0 のときは CI の下限・上限を入れ替えます。
- T0 と T1 が同じ値はエラーにします（`check_treatment_change`）。

## 効果がとても小さい単位のとき（`scaling`）

econml の因果フォレストの CI は、CATE の分散がおよそ 1e-10 以下になると、内部の固定のしきい値で
標準誤差が NaN になり、ATE の CI が点推定と同じ値（幅 0）になります（エラーも警告も出ません）。
Y の単位が小さい、または T の単位が大きいデータで起きます。

Y を定数 c 倍して推定し、結果を c で割って戻せば避けられます。部分線形モデルでは効果・CI がちょうど c 倍に
なるので、統計的な意味は変わりません。ノートブックでは `Y_SCALE` で指定し、表・図・標準出力は元の単位に
戻して表示します。戻し方は `scaling` モジュールの説明（効果は 1/c 倍、Y のモデルの MSE は 1/c² 倍、
決定係数・重要度・robustness value・E-value・p 値はそのまま）を参照してください。

| 関数 | 内容 |
|---|---|
| `scaling.check_scale` | 定数が正の有限な値か確かめる |
| `scaling.rescale_columns` | 表の効果の列を倍率倍する（`EFFECT_COLUMNS`・`OVB_EFFECT_COLUMNS`・`REFUTATION_EFFECT_COLUMNS`） |
| `EffectSummary.rescaled` | ATE・CI・CATE をまとめて倍率倍する |
| `NuisanceTuningResult.rescaled_mse` | チューニングの MSE を倍率倍する |

## 注意

- `estimator.dowhy` は参照のたびに新しいラッパーを作るので、`fit_dowhy_wrapper` の戻り値を使い回します。
- LightGBM のスレッド数の既定は 4 です（`DEFAULT_NUISANCE_PARAMS["n_jobs"]`）。全スレッドにすると、
  コア数の多い環境ではスレッドの取り合いで大幅に遅くなります。
- 反証テストは推定し直しを繰り返すので、ノートブックでは行数を減らして実行します。
