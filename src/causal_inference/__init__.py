"""causal_inference パッケージ（econml の DML による CATE 推定と感度分析）。

DAG の作成と estimand の同定は済んでいる前提で、推定から感度分析・可視化までの部品を提供する。
データの読み込みと前処理（inf・欠損の処理、Y/T/X/W への分割）は呼び出し側（ノートブック）で行い、
ここに渡す配列は欠損・inf を含まないこと（含む場合は `ValueError`）。

- `nuisance`: cross-fitting の分割と、nuisance モデル（E[Y|X,W]・E[T|X,W]）の Optuna チューニング
- `estimators`: econml の DML 推定器（LinearDML・CausalForestDML）の生成・学習・効果の取り出し
- `evaluation`: 真の CATE が分かる合成データでの推定精度の評価
- `sensitivity`: E-value、econml の omitted variable bias 感度分析、DoWhy の反証テスト
- `partial_r2`: 部分決定係数による感度分析（Cinelli & Hazlett 2020。robustness value と、
  観測された共変量による benchmark）
- `scaling`: 推定のために Y を定数倍したときの、結果の元の単位への戻し方
- `plots`: 推定結果と感度分析の図
- `output`: 図・表の保存
"""
