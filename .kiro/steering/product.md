# Product Overview

回帰データ（説明変数 `X`、目的変数 `y`）から、モデルの予測から大きく外れる**外れサンプル**を検出する Python ライブラリ。PyPI では `ensemble-outlier-sample-detection` として配布している。金子研の論文（README 参照）と参照実装 [hkaneko1985/ensemble_outlier_sample_detection](https://github.com/hkaneko1985/ensemble_outlier_sample_detection) を scikit-learn 風の API に再構成したもの。

対象ユーザーは、ケモインフォマティクスやマテリアルズインフォマティクスなど、少数サンプルで回帰モデルを作るデータサイエンティストや研究者。

## Core Capabilities

- **アンサンブルによるロバストな外れ値判定**: bootstrap した多数のサブモデルの予測中央値と実測値の差を、MAD ベースの 3σ 相当の基準で判定する
- **反復的な除外**: 外れと判定したサンプルを除いて再学習し、判定が変わらなくなるまで繰り返す
- **サブモデルの自動チューニング**: PLS（`n_components`）または RBF-SVR（`C` / `epsilon` / `gamma`）を、Optuna TPE で bootstrap ごとに最適化する
- **再現性**: `random_state` を固定すれば検出結果を再現できる

## Target Use Cases

- 回帰モデルを構築する前の前処理として、測定ミスや異常値を含むサンプルを特定・除外する
- 外れサンプルの候補を `outlier_support_`（boolean mask）で受け取り、可視化や人による精査に回す

## Value Proposition

- 単一モデルの残差ではなくアンサンブルの中央値を使うので、外れサンプル自体にモデルが引きずられにくい
- `BaseEstimator` を継承した `fit` → `*_support_` という sklearn 流儀の API で、既存の分析コードに組み込みやすい
- 公開 API は `EnsembleOutlierSampleDetector` 1 クラスだけで、学習コストが小さい

## Product Direction

依存ライブラリの古い API に依存しているため、最新の NumPy / scikit-learn では動作しない。現在の検出結果を golden テストで固定したうえで、依存環境のモダン化と scikit-learn estimator 規約への準拠を進めている（spec: `modernize-with-golden-tests`）。規約に合わせるための API 変更は破壊的変更として扱い、メジャーバージョンを上げる。

---
_Focus on patterns and purpose, not exhaustive feature lists_
_updated_at: 2026-10-19_
