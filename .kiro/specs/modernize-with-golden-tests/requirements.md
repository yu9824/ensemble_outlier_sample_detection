# Requirements Document

## Project Description (Input)
pytestで現状の動作を保存するテストを書く → モダンに依存環境を更新するおよびリファクタリング（docstringの充実やtyping, src-layout pyproject.toml, ruff.toml, mypy.ini etc., python-templateを参考にする)

### 誰が困っているか
- 本ライブラリ（PyPI: `ensemble-outlier-sample-detection`）のメンテナー。テスト・lint・型チェック・CI がないため、安全に変更できない
- 現在の Python / NumPy / scikit-learn / Optuna 環境でライブラリを使いたいユーザー

### 現状
- 削除済み・非推奨の API（`np.bool`、`mean_squared_error(..., squared=False)`、`trial.suggest_loguniform`）に依存しており、最新の NumPy / scikit-learn では動作しない
- パッケージングが `setup.py` + `requirements.txt` で、`py2app` 用の未使用分岐が残っている。flat layout を採用している
- テスト・ruff・mypy・CI がない。docstring と型ヒントも不十分
- `fit` が `None` を返す、コンストラクタ引数を上書きする、`max_iter` 内に収束しないと `outlier_support_` が未設定のまま `AttributeError` になる、など scikit-learn の estimator 規約から外れた挙動がある

### 変えたいこと
1. **現状動作の固定（先行作業）**: 旧依存環境（numpy<1.24 など）で現行コードを固定の `random_state` で実行し、`outlier_support_` などの出力を golden ファイルとして保存する。そのうえで、リファクタリング後も数値的な検出結果が golden と一致することを pytest で検証する
2. **依存環境のモダン化**: `pyproject.toml`（setuptools + setuptools-scm による動的バージョン）、src-layout、`ruff.toml`、`mypy.ini` へ移行する。`requires-python >= 3.10` とし、最新の NumPy / scikit-learn / Optuna で動作させる。参考実装はローカルの `~/projects/python-template`
3. **リファクタリング**: docstring と型ヒントを充実させる。既知の quirks は scikit-learn 準拠に修正する（`fit` は `self` を返す、コンストラクタ引数を変更しない、未収束時も結果属性を設定するなど）。これらは破壊的変更として扱い、メジャーバージョンを上げる
4. **CI**: python-template に倣って GitHub Actions（テスト・lint・型チェック、release、dependabot）を導入する

### スコープ外
- Sphinx ドキュメント、devcontainer / Docker

## Introduction

`ensemble-outlier-sample-detection` は、削除済みの依存 API に頼っているため最新の NumPy / scikit-learn では動かない。テストもないので安全に直せない状態にある。本 spec では、まず変更前の検出結果を golden として固定する。次に、その一致を確認しながら依存環境とパッケージングをモダン化し、scikit-learn の estimator 規約に合わせてリファクタリングする。規約に合わせる変更は破壊的変更として 1.0.0 でリリースする。

本文中のツール名（pytest、ruff、mypy、`pyproject.toml`、src-layout、setuptools-scm、GitHub Actions、PyPI trusted publishing）は、メンテナーが指定した制約である。

## Boundary Context

- **In scope**:
  - 変更前の検出結果の golden 化と、それを使った回帰テスト
  - パッケージングと依存環境のモダン化
  - scikit-learn estimator 規約への準拠（破壊的変更）
  - R² スコアの引数順の修正
  - `method='svr'` が必ず失敗する不具合（サブモデルの予測の形状）の修正
  - Optuna のログ設定に残る副作用の除去
  - docstring・型ヒント・README・変更履歴の整備
  - example notebook の更新
  - ruff / mypy の導入
  - CI、リリース（PyPI / TestPyPI への自動公開を含む）、依存更新ボット
- **Out of scope**:
  - Sphinx ドキュメント、devcontainer / Docker
  - 依存ライブラリの下限バージョンでのテスト
  - R² の修正以外のアルゴリズム変更（手法・評価指標の追加、判定基準の変更、性能の最適化）
  - Python 3.9 以前のサポート
  - conda-forge での配布
- **Adjacent expectations**:
  - golden の生成に使う旧依存環境は、メンテナーがローカルで用意する。CI では golden を生成しない
  - PyPI / TestPyPI 側の trusted publisher の登録と、GitHub 側の公開用 environment の設定は、メンテナーが手作業で行う。本 spec が扱うのはワークフロー側だけ
  - 規約は python-template（`~/projects/python-template`）に従う。ただしライセンスは Apache License 2.0 のまま変えない

## Requirements

### Requirement 1: 変更前の検出結果の golden 化
**Objective:** As a メンテナー, I want 変更を加える前の検出結果を golden として保存したい, so that リファクタリングや依存更新で結果が変わっていないことを確かめられる

#### Acceptance Criteria
1. The golden 生成手順 shall 本 spec によるコード変更を加える前のコードを、2021 年当時の依存環境（Python 3.8、NumPy 1.20、scikit-learn 0.24、pandas 1.2、Optuna 2.8 に固定したもの）で実行して golden を生成する
2. The golden 生成手順 shall 手法（`pls`、`svr`）と評価指標（`r2`、`mse`、`mae`、`rmse`）のすべての組み合わせを golden の対象にする
3. When 開発者が golden 生成手順を実行する, the golden 生成手順 shall 構成ごとに outlier mask、inlier mask、収束までに実行した反復回数を golden ファイルに保存する
4. The golden 生成手順 shall 構成ごとに、入力データ、`random_state` を含むパラメータ、生成時の Python と依存ライブラリのバージョンを golden とあわせて記録する
5. The golden 生成手順 shall 旧依存環境の作り方と実行手順を文書化し、第三者が同じ golden を再生成できるようにする
6. If 変更前のコードがある構成で `max_iter` 内に収束せず、結果を記録できない, the golden 生成手順 shall 収束するようにパラメータを調整した構成で golden を生成し、調整した内容と理由を記録する
7. The テストスイート shall 参照データとして example のデータセットを使い、原データの出典表記とライセンス表記を残す
8. Where 手法が `svr` である, the golden 生成手順 shall 変更前のコードでは SVR が動かないため、サブモデルの予測の形状を揃える最小限の修正だけを加えたコードで golden を生成し、加えた修正の内容を記録する
9. The golden 生成手順 shall SVR 用の最小修正を加えても `pls` の検出結果が変わらないことを確かめる
10. The golden 生成手順 shall 2021 年当時の環境の golden とは別に、変更前のコードに最新の依存環境で動かすための最小限の互換修正だけを加え、最新の依存環境で実行した結果を「最新環境の golden」として記録する。記録するのは、構成ごとの outlier mask、収束までに実行した反復回数、最後の反復におけるアンサンブル予測の中央値（サンプルごと）と、加えた互換修正の内容

### Requirement 2: golden を使った回帰テスト
**Objective:** As a メンテナー, I want 変更のたびに golden との一致を自動で確かめたい, so that 検出結果を意図せず変えてしまうことを防げる

#### Acceptance Criteria
1. When 開発者がテストスイートを実行する, the テストスイート shall golden の各構成を現在のコードで実行し、outlier mask と収束までの反復回数が golden と完全に一致することを検証する
2. If golden と一致しない構成がある, the テストスイート shall 失敗を報告し、該当する構成の名前と結果が異なるサンプルを表示する
3. The テストスイート shall テストを実行しても golden ファイルを書き換えない
4. If 不一致の原因が本パッケージのコード変更にある, the 開発プロセス shall golden を更新せずにコードを修正する
5. If 不一致の原因が依存ライブラリの更新だと特定できた, the 開発プロセス shall メンテナーの承認を得た場合に限って golden を更新し、原因、依存ライブラリのバージョン、承認の記録を残す
6. When 同じデータと同じ整数の `random_state` で `fit` を 2 回実行する, the EnsembleOutlierSampleDetector shall 2 回とも同じ検出結果を返す
7. When 同じデータを列名付きの表形式と配列のどちらで与えても, the EnsembleOutlierSampleDetector shall 同じ検出結果を返す
8. When 開発者がテストスイートを実行する, the テストスイート shall 各構成について、outlier mask と反復回数が最新環境の golden と完全に一致し、最後の反復におけるアンサンブル予測の中央値が許容誤差の範囲で一致することを検証する

### Requirement 3: 依存環境とパッケージングのモダン化
**Objective:** As a 利用者, I want 現在の Python と依存ライブラリの環境にそのままインストールして使いたい, so that 古い環境を維持しなくて済む

#### Acceptance Criteria
1. The パッケージ shall 要件として Python 3.10 以上を宣言する
2. The パッケージ shall 対応する各 Python バージョンにインストールできる最新の安定版の NumPy、pandas、scikit-learn、Optuna、tqdm で動作する
3. While テストスイートを実行している, the パッケージ shall 自身のコードを原因とする非推奨警告（`DeprecationWarning`、`FutureWarning`）を出さない
4. The パッケージ shall ビルド設定とメタデータを `pyproject.toml` に一元化し、`setup.py` と `requirements.txt` を廃止する
5. The パッケージ shall ソースを src-layout で配置する
6. The パッケージ shall バージョンを git タグから決め、実行時には `__version__` で参照できるようにする
7. The パッケージ shall 配布名 `ensemble-outlier-sample-detection` と import 名 `ensemble_outlier_sample_detection` を変えない
8. When 利用者が `from ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector` を実行する, the パッケージ shall これまでどおり検出器クラスを提供する
9. The パッケージ shall メタデータにライセンス（Apache License 2.0）、作者、プロジェクト URL を含め、URL は実際のリポジトリ（`github.com/yu9824/ensemble_outlier_sample_detection`）を指す
10. The パッケージ shall 型情報を同梱していることを示す PEP 561 のマーカーを配布物に含める
11. When 開発者がビルドを実行する, the パッケージ shall sdist と wheel を生成し、wheel には example とテストを含めない
12. The パッケージ shall テスト用と開発用の追加依存を、オプショナル依存として宣言する
13. The リポジトリ shall ビルド成果物とツールのキャッシュをバージョン管理の対象から外す
14. The パッケージ shall 本変更を含む最初のリリースをバージョン 1.0.0 とする

### Requirement 4: scikit-learn estimator 規約への準拠
**Objective:** As a 利用者, I want 検出器を scikit-learn の他の estimator と同じ作法で扱いたい, so that `clone` や再学習、メソッドチェーンを含む既存のワークフローに組み込める

#### Acceptance Criteria
1. When 利用者が `fit(X, y)` を呼ぶ, the EnsembleOutlierSampleDetector shall 学習を終えた自身のインスタンスを返す
2. The EnsembleOutlierSampleDetector shall `fit` の前後で、`get_params()` が返すコンストラクタ引数の値を変えない
3. When 学習済みのインスタンスで `fit` を再び呼ぶ, the EnsembleOutlierSampleDetector shall エラーを出さずに与えられたデータで学習し直し、学習結果を上書きする
4. When `clone` で複製したインスタンスを元のインスタンスと同じデータで `fit` する, the EnsembleOutlierSampleDetector shall 元のインスタンスと同じ検出結果を返す
5. The EnsembleOutlierSampleDetector shall これまでどおり、`method` に `pls`、`PLS`、`svr`、`SVR` を、`metric` に `r2`、`mse`、`mae`、`rmse` を受け付ける
6. If `method` か `metric` に未対応の値が指定される, the EnsembleOutlierSampleDetector shall `fit` の時点で `ValueError` を送出し、受け付ける値をメッセージで示す
7. If `max_iter` 回反復しても判定が収束しない, the EnsembleOutlierSampleDetector shall 最後の反復の判定結果を `outlier_support_` と `inlier_support_` に設定し、収束しなかったことを `ConvergenceWarning` で通知する
8. When `fit` が完了する, the EnsembleOutlierSampleDetector shall 実際に実行した反復の回数を `n_iter_` に設定する（従来の 0 始まりのインデックスから回数に変える）
9. The EnsembleOutlierSampleDetector shall 非標準の属性 `n_iter_finished` を廃止する
10. When `fit` が完了する, the EnsembleOutlierSampleDetector shall 学習に使った特徴量の数を `n_features_in_` に設定し、列名付きの入力を受け取った場合は列名を `feature_names_in_` に設定する
11. The EnsembleOutlierSampleDetector shall bootstrap した中間データ、スケーラー、評価関数など学習途中の一時的な状態を、`fit` の後に公開属性として残さない
12. If `fit` の前に `get_outlier_support()` か `get_inlier_support()` が呼ばれる, the EnsembleOutlierSampleDetector shall 未学習であることを示す `NotFittedError` を送出する
13. If `X` と `y` のサンプル数が違う、`y` が与えられない、または入力に欠損値や無限大が含まれる, the EnsembleOutlierSampleDetector shall `ValueError` を送出する

### Requirement 5: 実行時の副作用と進捗表示
**Objective:** As a 利用者, I want `fit` が自分の実行環境のグローバルな設定を変えないようにしたい, so that 同じプロセスで Optuna を使う他のコードに影響が出ない

#### Acceptance Criteria
1. While `fit` を実行している, the EnsembleOutlierSampleDetector shall Optuna の試行ごとのログを出力しない
2. When `fit` が正常終了するか例外で中断する, the EnsembleOutlierSampleDetector shall Optuna のグローバルなログ設定を `fit` を呼ぶ前の状態に戻す
3. Where `progress_bar=True` が指定されている, the EnsembleOutlierSampleDetector shall 現在の反復と総反復数を進捗バーに表示し、終了したら完了を表示して閉じる
4. While Jupyter のカーネル上で `progress_bar=True` で実行している, the EnsembleOutlierSampleDetector shall notebook 向けの進捗バーを表示する
5. If 反復が収束しないまま終了する, the EnsembleOutlierSampleDetector shall 進捗バーをエラーなく閉じる
6. Where `progress_bar=False` が指定されている, the EnsembleOutlierSampleDetector shall 進捗バーを表示しない
7. The パッケージ shall モジュールを直接実行しても対話デバッガを起動しない

### Requirement 6: R² スコアの引数順の修正
**Objective:** As a 利用者, I want `metric='r2'` のときに R² が正しい定義で計算されてほしい, so that ハイパーパラメータが本来の評価基準で選ばれる

#### Acceptance Criteria
1. The 開発プロセス shall Requirement 2 の golden 一致を確認した後で、R² の修正を独立した変更として適用する
2. When `metric='r2'` でハイパーパラメータを探索する, the EnsembleOutlierSampleDetector shall 実測値と予測値を定義どおりの順序で渡して R² を計算する
3. When R² の修正を適用する, the 開発プロセス shall 変更前のコードに R² の引数順の修正を加えたものから、`r2` を使う構成の golden（2021 年当時の環境と最新環境の両方）だけを再生成し、更新理由を記録する
4. While R² の修正を適用した後, the テストスイート shall `mse`、`mae`、`rmse` を使う構成の golden を更新しないまま一致させ続ける

### Requirement 7: docstring・型ヒント・利用者向けドキュメント
**Objective:** As a 利用者, I want API の仕様と移行方法をコードとドキュメントから読み取りたい, so that 1.0.0 への移行や日々の利用で迷わない

#### Acceptance Criteria
1. The パッケージ shall すべての公開クラスと公開メソッドに NumPy スタイルの docstring を付け、引数（取りうる値と既定値）、戻り値、学習後の属性、送出する例外を書く
2. The パッケージ shall 検出器の docstring にアルゴリズムの概要と参照論文を書く
3. The パッケージ shall 公開 API と内部の関数・メソッドのすべてに型ヒントを付ける
4. The README shall 新しい API での使い方、対応する Python のバージョン、インストール方法を書く
5. The 変更履歴 shall 1.0.0 の破壊的変更と移行方法を書く。対象は、`fit` の戻り値、`n_iter_` の意味の変更、`n_iter_finished` の廃止、未対応の値に対して送出する例外の変更、Python 3.9 以前のサポート終了、`metric='r2'` で検出結果が以前と変わり得ること、`method='svr'` が動くようになったこと（バグ修正）
6. When 新しい依存環境で example notebook を先頭から実行する, the example notebook shall エラーなく最後まで実行でき、新しい API を使った出力に更新されている

### Requirement 8: 静的解析
**Objective:** As a メンテナー, I want コードスタイルと型の問題を機械的に見つけたい, so that レビューの負担を減らして品質を保てる

#### Acceptance Criteria
1. The リポジトリ shall lint と format の設定（1 行 79 文字、NumPy スタイルの docstring 規約）を `ruff.toml` に置く
2. The リポジトリ shall 型チェックの設定を `mypy.ini` に置く
3. When 開発者が lint と format の検査を実行する, the パッケージのソースとテスト shall 違反なしで検査を通る
4. When 開発者が型チェックを実行する, the パッケージのソース shall エラーなしで型チェックを通る

### Requirement 9: CI とリリース
**Objective:** As a メンテナー, I want 検査とリリースを自動化したい, so that 手作業のミスなく品質を保ち、PyPI へ公開できる

#### Acceptance Criteria
1. When main ブランチに push されるか、pull request が作成または更新される, the CI shall Python 3.10 から 3.14 の各バージョンでテストスイートを実行する
2. When main ブランチに push されるか、pull request が作成または更新される, the CI shall lint・format の検査と型チェックを実行する
3. If いずれかの検査が失敗する, the CI shall 失敗を報告する
4. The CI shall テストジョブを 1 つあたり 10 分以内で完了させる
5. When `v<major>.<minor>.<patch>` 形式のバージョンタグが push される, the リリースワークフロー shall sdist と wheel をビルドし、署名を付けて GitHub Release に添付する
6. When 安定版か rc のバージョンタグが push される, the リリースワークフロー shall trusted publishing で PyPI に公開する
7. When プレリリースを含むバージョンタグが push される, the リリースワークフロー shall trusted publishing で TestPyPI に公開する
8. The 依存更新ボット shall Python の依存と GitHub Actions の更新を週 1 回提案する
