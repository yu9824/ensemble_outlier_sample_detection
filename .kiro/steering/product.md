# Product Overview

回帰データ（説明変数 `X`、目的変数 `y`）から、モデルの予測から大きく外れる**外れサンプル**を検出する Python ライブラリ。PyPI では `ensemble-outlier-sample-detection` として配布している。金子の論文（Kaneko 2018, Chemometrics and Intelligent Laboratory Systems 177, 74–82）と参照実装 [hkaneko1985/ensemble_outlier_sample_detection](https://github.com/hkaneko1985/ensemble_outlier_sample_detection) を、scikit-learn の estimator として再構成したもの。

対象ユーザーは、ケモインフォマティクスやマテリアルズインフォマティクスなど、少数サンプルで回帰モデルを作るデータサイエンティストや研究者。

## Core Capabilities

- **アンサンブルによるロバストな外れ値判定**: bootstrap した多数のサブモデルの予測中央値と実測値の差を、MAD ベースの 3σ 相当の基準（`3 * 1.4826 * MAD`）で判定する
- **反復的な除外**: 外れと判定したサンプルを除いて学習し直し、判定が変わらなくなるまで繰り返す。`max_iter` に達しても収束しなければ、最後の判定を結果にして `ConvergenceWarning` を出す
- **サブモデルの自動チューニング**: PLS（`n_components`）または RBF-SVR（`C` / `epsilon` / `gamma`）を、bootstrap ごとに Optuna TPE で最適化する。評価指標は `r2` / `mse` / `mae` / `rmse`
- **再現性**: 整数の `random_state` を固定すれば、検出結果を再現できる

## Target Use Cases

- 回帰モデルを作る前の前処理として、測定ミスや異常値を含むサンプルを特定して除外する
- 外れサンプルの候補を `outlier_support_` / `inlier_support_`（boolean mask）で受け取り、可視化や人による精査に回す

## Value Proposition

- 単一モデルの残差ではなくアンサンブルの中央値を使うので、外れサンプル自体にモデルが引きずられにくい
- scikit-learn の規約に従う（`fit` は self を返す、`get_params()` は変わらない、`clone` と再学習ができる、`n_features_in_` / `feature_names_in_` を持つ）ので、既存の分析コードに組み込みやすい
- 公開 API は `EnsembleOutlierSampleDetector` の 1 クラスだけで、覚えることが少ない
- fit の間だけ Optuna のログを抑え、終わったら利用者のログ設定に戻すので、同じプロセスの他のコードに影響しない

## Product Direction

1.0.0 で、最新の Python（3.10〜3.14）と依存ライブラリへの対応、scikit-learn 規約への準拠、R² の引数順の修正、`method="svr"` の修正を、破壊的変更として出す（spec: `modernize-with-golden-tests`。変更点は `CHANGELOG.md`、公開手順は `HOW_TO_RELEASE.md`）。検出結果は golden テストで固定しているので、今後の変更でも結果が意図せず変わらないことを確かめられる。アルゴリズム自体（判定基準や探索範囲）を変えるときは、golden の更新ルール（`tests/golden/README.md`）に従う。

---
_Focus on patterns and purpose, not exhaustive feature lists_
_updated_at: 2026-10-20（1.0.0 の実装完了にあわせて、移行期の記述を今の状態に更新）_
