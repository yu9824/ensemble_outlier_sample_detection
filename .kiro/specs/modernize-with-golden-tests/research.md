# Research & Gap Analysis: modernize-with-golden-tests

_実施日: 2026-10-19 / 対象: requirements.md（Requirement 1〜9）_

## 1. 現状調査

### 1.1 資産と構造
- ロジックは `ensemble_outlier_sample_detection/ensemble_outlier_sample_detection.py`（263 行）の 1 ファイルにまとまっている。中身は公開クラス `EnsembleOutlierSampleDetector`、内部クラス `Objective`、デバッグ用の `__main__` ブロック
- `__init__.py` は `import *` で再エクスポートし、メタデータを単一行の文字列リテラルで持つ。`setup.py` はこれを正規表現で抜き出している
- `example/example.ipynb` は Python 3.8.8 のカーネルで実行された出力（約 5.5MB）を含む。notebook 自身のセルでも `mean_squared_error(..., squared=False)`（cell 11 / 21）を使っている。検出器は既定値（`n_estimators=100`、`max_iter=30`）に `metric='rmse'` を指定して呼んでいる
- 参照データ `example/example_chache.csv` は 100 サンプル × 10 特徴量。先頭に UTF-8 BOM と、出典を書いた `#` コメント行がある。`comment='#'` を付ければ pandas 1.2 でも 2.x でも正しく読める（実測済み）
- テスト、lint、型チェック、CI、`.gitignore` はどれもない

### 1.2 リファクタリングで保たなければならない暗黙の挙動（golden 一致に効くもの）
| # | 挙動 | 位置 |
|---|------|------|
| P1 | RNG から seed を引く順序。bootstrap ごとに `randint(2**32)` を 2 回、resample → TPESampler の順で引く | L124, L149 |
| P2 | `max_components` は bootstrap をまたいで**累積的な最小値**になる（`self.max_components = min(self.max_components, rank)` で自身を上書きするため、一度下がると戻らない）。Optuna の試行回数 `min(100, max_components)` もこの値に従う | L143, L151 |
| P3 | MAD は、前回 inlier だったサンプルについて、全サブモデルの予測と「サンプルごとの中央値の、さらに中央値」との差から計算する | L177 |
| P4 | 収束判定。mask が前回と同じ、**または** inlier が 0 件になったら収束とみなす | L186 |
| P5 | Optuna の `best_params` は、同じスコアの試行が複数あれば最初の試行を採る（Optuna 側の仕様に従う） | L154 |
| P6 | 評価関数の引数は `metric(y_pred, y_true)` の順。Requirement 6 で修正するまでは保つ | L250 |

## 2. 実測による動作確認

読み取り専用で既存の conda 環境を使い、現行コードをそのまま実行した（環境には何も変更を加えていない）。

| 環境 | Python | numpy | scikit-learn | pandas | optuna | tqdm |
|------|--------|-------|--------------|--------|--------|------|
| 旧（`conductivity-prediction-old`） | 3.8.20 | 1.20.1 | 0.24.1 | 1.2.4 | 2.8.0 | 4.59.0 |
| 最新（`yikit-dev`） | 3.12.15 | 2.5.3 | 1.9.1 | 2.3.3 | 5.0.0 | 4.70.1 |

### 2.1 結果（`random_state=334`、`n_estimators=3`、`max_iter=5`）
| 構成 | 旧環境 | 最新環境 |
|------|--------|----------|
| pls / r2 | OK: outlier 14 件、`n_iter_finished=2` | **同一** |
| pls / mse | OK: outlier 15 件、`n_iter_finished=4` | **同一** |
| pls / mae | OK: outlier 15 件、`n_iter_finished=2` | **同一** |
| pls / rmse | OK: outlier 15 件、`n_iter_finished=4` | FAIL: `squared` 引数がない（TypeError） |
| svr / 全 metric | **FAIL**: `Item wrong length 100 instead of 300`（L174） | **FAIL**: `Expected 2D array, got 1D array`（L250） |

**既定値（`n_estimators=100`、`max_iter=30`）での PLS の比較（R1、調査済み）**。rmse については、最新環境だけ probe 側で `squared=` の互換関数を差し込んで実行した。

| 構成 | 旧環境 | 最新環境 | 実行時間（旧 / 最新） |
|------|--------|----------|------------------------|
| pls / r2 | outlier 14 件、`n_iter_finished=6`、mask `ff1254e312` | **同一** | 27s / 32s |
| pls / mse | outlier 15 件、`n_iter_finished=2`、mask `a634f065c7` | **同一** | 12s / 13s |
| pls / mae | 同上 | **同一** | 12s / 13s |
| pls / rmse | 同上 | **同一** | 12s / 14s |

- DataFrame で渡しても ndarray で渡しても、両方の環境で結果は同じだった（Requirement 2.7 の裏付け）
- mask の値は `np.packbits` を SHA-1 でハッシュし、先頭 10 桁を示している

**SVR の最小修正の検証（S1）**。scratch のコピーで L163 を `inverse_transform(np.reshape(y_pred, (-1, 1)))` に変えただけで、旧環境で svr × 4 metric がすべて動いた（`n_estimators=3` で outlier 0 件、約 1 秒/サブモデル/反復）。同じ修正のもとで、PLS の結果は変更前と同じだった（Requirement 1.9 の裏付け）。目的関数の中の `inverse_transform(y_pred_oof)` は、旧 scikit-learn が 1 次元の入力を通すため、修正しなくても旧環境では動く。最新環境では、リファクタリングの中で目的関数側の形状も揃える必要がある。

### 2.2 分かったこと
- **F1: SVR は現行コードの時点ですでに動かない（旧環境でも同じ）**。`SVR.predict` は 1 次元の配列を返す。旧 scikit-learn の `StandardScaler.inverse_transform` は 1 次元をそのまま通すが、その後 `np.hstack` が各サブモデルの予測を縦に連結して長さ `n × n_estimators` の 1 次元配列になり、boolean index の長さが合わなくなる。新しい scikit-learn では、それより前の `inverse_transform` が 1 次元の入力を拒否する。**Requirement 1.2 の「svr の golden」は、現行コードからは生成できない**
- **F2: PLS は旧環境と最新環境で結果が完全に一致する**。Optuna 2.8 と 5.0、scikit-learn 0.24 と 1.9、numpy 1.20 と 2.5 で、mask と反復回数が同じだった。依存ライブラリの違いで golden が一致しなくなるリスクは、PLS については低い
- **F3: 最新環境で問題になる API は、実際には `squared=` だけ**。`np.bool` は NumPy 2.0 で `np.bool_` の別名として復活したので、失敗するのは NumPy 1.24〜1.26 だけ。`suggest_loguniform` は Optuna 5.0 でも `FutureWarning` 付きで動く
- **F4: 出ている警告**: `DeprecationWarning`（`np.bool`、旧 NumPy のみ）、`FutureWarning`（`suggest_loguniform`）、`DataConversionWarning`（SVR に列ベクトルの y を渡している）。Requirement 3.3 を満たすには、どれも解消する必要がある
- **F5: 未収束時の進捗バー**: `n_iter_finished` が設定されないまま L196 で参照され、`AttributeError` になる（steering に記載済みの quirk を確認した）
- **F6: notebook の依存**: notebook 自身のコードが `squared=False` と、フォント `Helvetica` の指定に依存している。Requirement 7.6 を満たすには、notebook のコードも直す必要がある

## 3. Requirement-to-Asset Map

| Req | 既存資産 | ギャップ | メモ |
|-----|----------|----------|------|
| 1 golden 化 | なし | **Missing** / **Constraint (F1)** | PLS は旧環境で生成できる。SVR は現行コードが壊れているため、要件の見直しが必要 |
| 2 回帰テスト | なし | Missing | 保存形式、比較方法、失敗時の差分表示を新しく作る |
| 3 パッケージング | `setup.py`、`requirements.txt`、`__init__.py` のメタデータ | Missing | python-template の `pyproject.toml` を流用できる。`__url__` の誤りを直す |
| 4 sklearn 規約 | `BaseEstimator`、`check_X_y`、`check_is_fitted` | Missing / Constraint (P1, P2) | 引数を書き換えないようにしても、P2 の累積最小値は局所変数で再現する必要がある |
| 5 副作用・進捗 | `disable_default_handler`、tqdm の手動切り替え | Missing / Unknown | ログ設定を退避して戻す方法は要調査（R5） |
| 6 R² の修正 | L250 | Constraint (P6) | golden の後に、独立した変更として入れる |
| 7 ドキュメント | 簡易な docstring、README | Missing / Constraint (F6) | notebook のコードも直す必要がある |
| 8 静的解析 | なし | Missing | python-template の `ruff.toml` と `mypy.ini` を流用できる |
| 9 CI・リリース | なし | Missing | python-template の CI / release / dependabot を流用できる。Docker 関連は除く |

## 4. 実装アプローチの選択肢

### Option A: 既存モジュールを拡張する（1 ファイルのまま）
- 実装ファイルを `src/ensemble_outlier_sample_detection/` に移し、同じファイルの中で書き換える。`Objective` の後方参照は引数で渡す形に変える
- ✅ 差分を追いやすく、golden との対応が分かりやすい
- ❌ 検証、目的関数、反復ループ、進捗表示が 1 ファイルに残り、責務を分けにくい

### Option B: 新しいコンポーネントに分割する
- 公開クラス、目的関数（PLS / SVR の探索空間）、評価関数の対応表、ログや進捗の制御を、それぞれ private モジュールに分ける
- ✅ テストしやすく、型も付けやすい。副作用の制御を 1 か所に集められる
- ❌ golden との一致を崩さずに分割する手間がかかる（P1〜P5）。規模に比べてファイルが増えすぎるおそれがある

### Option C: ハイブリッド（段階的に進める）【推奨候補】
1. **Phase 0**: 変更前のコードで、旧環境を使って golden を生成する（PLS × 4 metric）
2. **Phase 1**: パッケージングだけを移行する（src-layout、`pyproject.toml`、ruff、mypy）。ロジックは変えず、API 置き換えに必要な最小限の修正（`squared=`）だけ入れて、最新環境で golden 一致を確かめる
3. **Phase 2**: sklearn 規約への準拠、副作用の除去、型ヒントと docstring。公開モジュールは 1 つに保ち、補助処理は必要な分だけ private 関数・モジュールに切り出す
4. **Phase 3**: R² の修正と、r2 構成の golden の再生成
5. **Phase 4**: CI、リリース、README、変更履歴、notebook
- ✅ どの段階でも golden で安全を確かめられる。不一致が出たときに、どの変更が原因かを切り分けやすい
- ❌ 段階を管理する手間が増える

### golden 生成環境の選択肢
- **G1: 2021 年当時の環境（Python 3.8 系のピン留め）**: 公開されていたときの挙動にいちばん忠実。ユーザーの指示（numpy<1.24 などの旧依存環境）にも合う。F2 から、PLS では最新環境との差は出ないと見込める
- **G2: 現行コードが動く範囲で最も新しい環境**（numpy 2.x + scikit-learn 1.5 系 + 最新の Optuna）: 最新環境との差がさらに小さい。ただし「旧依存環境」と呼べるかは曖昧
- いずれの場合も、他プロジェクトの環境（`conductivity-prediction-old`）を直接使うのではなく、ピン留めした環境定義から作り直せるようにする（Requirement 1.5）

### SVR の扱いの選択肢（**要件の判断が必要**）
- **S1: 最小修正をしてから golden を生成する**: 予測を 2 次元に揃える修正だけを先に入れ、そのコードを旧環境で動かして SVR の golden を作る。SVR の「本来意図された挙動」を固定できる
- **S2: SVR は golden の対象から外す**: リファクタリングの中で修正し、性質テスト（意図的に入れた外れ値を検出できるか、など）で確かめる。修正後の結果を、以後の回帰テスト用の基準値として新たに記録する
- **S3: SVR のサポートをやめる**: 破壊的変更の範囲が広がる。推奨しない

## 5. 工数とリスク
- **工数: M（3〜7 日）**。コードは小さいが、golden の仕組み、パッケージング、リファクタリング、CI とリリース、ドキュメントと notebook の 5 系統の作業がある
- **リスク: Medium**。F2 から、PLS の数値的な等価性は高い見込み。一方で、SVR には比較できる基準値がない（F1）。golden を一致させたまま P1 と P2 を保つリファクタリングには注意が要る。技術的に未知のものはない

## 6. Research Needed（設計フェーズへ引き継ぐ）
- **R1**: 既定値（`n_estimators=100`、`max_iter=30`）の PLS で、旧環境と最新環境の結果が一致するか → **調査済み、完全に一致**（§2.1）
- **R2**: SVR を直したときの意味論 → **調査済み**。参照実装（hkaneko1985/`ensemble_outlier_detection.py`）では、各サブモデルの予測を 1 次元に平坦化し、1 列の DataFrame にしてから `pd.concat(axis=1)` で「サブモデル 1 つにつき 1 列」に並べている。したがって、予測を `(n, 1)` に揃えて横に連結するという最小修正（S1）は、参照実装の意図と同じになる。P3 の MAD の式（全予測と、サンプルごとの中央値の中央値との差）も参照実装と同じなので、バグではなく意図された挙動である。なお、参照実装はハイパーパラメータを GridSearch と 2-fold CV で決めているが、本ライブラリは Optuna と `cv` 引数を使う独自の設計で、今回の変更対象ではない
- **R3**: CI のテストジョブを 10 分以内に収めるため、golden の各構成で使う `n_estimators` と `max_iter` の値。SVR は 1 bootstrap あたり 100 試行 × 5 fold かかる
- **R4**: 依存の下限バージョン。scikit-learn 1.8 以降は Python 3.11 以上が必要（[1.8.0 リリース告知](https://discuss.scientific-python.org/t/ann-scikit-learn-1-8-0-has-been-released/2196)、[インストール手順](https://scikit-learn.org/1.9/install.html)）なので、Python 3.10 で CI を回すと scikit-learn は 1.7 系になる。下限はこれより低く置く必要がある。ほかに確認すること:`root_mean_squared_error`（scikit-learn 1.4 以上）、`validate_data`（1.6 以上）、Python 3.10 に対応する NumPy・pandas・Optuna の最終版、`suggest_float(log=True)` と `suggest_loguniform` で乱数系列が同じかどうか
- **R5**: Optuna のログ設定を退避して戻す方法（`get_verbosity` / `set_verbosity`、default handler の状態）
- **R6**: notebook かどうかの判定を、手書きの `ipykernel` 判定から `tqdm.auto` に置き換えてよいか（Requirement 5.4 との整合）
- **R7**: pandas 3.0（Copy-on-Write）で PLS の結果が変わらないか。CI のマトリクスで確認する
- **R8**: PyPI / TestPyPI の trusted publishing に必要な設定（environment 名、権限）と、python-template にある action のバージョン

## 7. 要件への反映（差分分析の後にユーザーが決めたこと）
- SVR は S1（最小修正をしてから golden を生成する）を採る → Requirement 1.8 と 1.9 を追加し、Boundary と Requirement 7.5 に追記した
- golden の生成環境は G1（2021 年当時の環境）を採る → Requirement 1.1 にバージョンを書いた
- Requirement 3.2 を「各 Python バージョンにインストールできる最新の安定版」に改めた（R4 の制約による）

---

# Design Discovery（kiro-spec-design, light discovery）

## Summary
- **Feature**: `modernize-with-golden-tests`
- **Discovery Scope**: Extension（既存の小さなライブラリを、パッケージング・テスト・CI ごと刷新する）
- **使ったスキルとガイド**: kiro-spec-design の rules（design-discovery-light、design-synthesis、design-review-gate、design-principles）。domain 固有のスキルはない
- **Key Findings**:
  1. SVR は、どの構成でも outlier 0 件で初回に収束する（`n_estimators` 3〜20、`random_state` 0 / 334、`cv` 2 / 5 で確認）。そのため mask だけでは SVR の退行を検出できない
  2. SVR のアンサンブル予測の中央値は、Optuna 2.8 と 5.0 で大きく異なる（相対差は最大 69%）。PLS は 1e-15 以下の差で一致する
  3. `suggest_loguniform` の実装は `suggest_float(log=True)` を呼んでいるだけなので、置き換えても乱数系列は変わらない

## Research Log

### SVR の環境差と検出力
- **Context**: R3（実行時間）と、SVR の golden として何を記録すべきか
- **Sources Consulted**: scratch のコピーに最小修正を当てた変更前コードを、旧環境（Optuna 2.8）と最新環境（Optuna 5.0）で実測した
- **Findings**:
  - SVR の mask はすべて inlier で一致した（`022f36e301`）。予測の中央値は一致しない。RBF-SVR は外れサンプルにも当てはまるので、残差が `3 × 1.4826 × MAD` を超えない
  - 実行時間は、旧環境・最新環境ともに約 1 秒/サブモデル/反復（100 試行 × 5 fold、100 サンプル）。PLS は既定値（100 サブモデル）で 12〜32 秒/構成
- **Implications**: 2021 年環境の golden（mask と反復回数）に加えて、最新環境の golden（mask、反復回数、予測の中央値を許容誤差付きで比較）を持つ 2 層構成にする。要件 1.10 と 2.8 はユーザーの承認を得て追加した

### Optuna のログ制御
- **Sources Consulted**: Optuna 5.0.0 の `optuna.logging` の公開 API を調べた
- **Findings**: `get_verbosity` と `set_verbosity` で状態を読み書きできる。`disable_default_handler` には状態を取得する関数がない。既定の verbosity は INFO（20）で、試行ごとのログは INFO で出る
- **Implications**: `fit` の間だけ verbosity を WARNING にし、`finally` で元の値に戻す。handler には触らない

### tqdm の自動切り替え
- **Findings**: notebook の外では `tqdm.auto.tqdm` は通常の（asyncio 版の）tqdm になる。Jupyter のカーネル上では notebook 版を選ぶ。`disable=True` を渡すと何も出力しない
- **Implications**: `'ipykernel' in sys.modules` による手書きの判定をやめて、`tqdm.auto` に任せる。`progress_bar=False` の場合は `disable=True` にして、コードの経路を 1 つにまとめる

### scikit-learn の入力検証
- **Findings**: `sklearn.utils.validation.validate_data(estimator, X, y, ...)` は 1.6 以降の公開 API で、`n_features_in_` と `feature_names_in_` を設定する。既定で NaN と inf を拒否する
- **Implications**: 依存の下限を `scikit-learn>=1.6` にする。1.6 は Python 3.9〜3.13 に、1.7 は 3.10 以上に対応するので、Python 3.10 でも条件を満たせる

### Python のバージョンごとに入る依存の版
- **Findings**（各パッケージの Requires-Python）: numpy 2.5 は 3.12 以上、scikit-learn 1.9 は 3.11 以上（1.8 以降が 3.11 以上）、scipy 1.18 は 3.12 以上。Optuna 5.0 は 3.9 以上、pandas 2.3 は 3.9 以上
- **Implications**: CI の Python 3.10 ではやや古い numpy / scikit-learn が入るが、Optuna は全バージョンで同じものが入る。SVR の予測差の主な原因は Optuna なので、マトリクス内では最新環境の golden の許容誤差に収まると見込む。収まらなければ要件 2.5 の手順で対応する

### setuptools-scm と CI の checkout
- **Findings**: setuptools-scm 8 以降は `version_file` を推奨し、`write_to` は非推奨になっている。タグからバージョンを決めるには、タグとその履歴が checkout されている必要がある
- **Implications**: `version_file` を使う。リリースの build ジョブでは `fetch-depth: 0` にする

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| A 1 ファイルで拡張 | 実装ファイルを src に移し、その中で書き換える | 差分が追いやすい | 検証・探索・副作用の制御が 1 ファイルに混在する | |
| B 細かく分割 | validation、objective、loop、runtime などを個別の private モジュールにする | テストしやすい | 規模に比べてファイルが増えすぎる | |
| **C 段階的・軽い分割** | 公開モジュールは 1 つのまま、探索まわり（`_submodels`）と実行時の副作用（`_runtime`）だけを切り出す。Phase 0〜4 で進める | 各段階を golden で守れる。責務の境界がはっきりする | 段階の管理が必要 | **採用** |

## Design Decisions

### Decision: golden を 2 層にする（2021 年環境と最新環境）
- **Context**: 要件 1.1、2.1 と、SVR の検出力（Key Finding 1、2）
- **Alternatives Considered**: 1) 2021 年環境の mask だけ　2) すべての予測を比較する　3) 2 層にする
- **Selected Approach**: 1 つの case ファイルに `legacy` ブロックと `modern` ブロックを持たせる。legacy は mask と反復回数を完全一致で比べる。modern は mask と反復回数を完全一致、予測の中央値を `rtol=1e-7`、`atol=1e-10` で比べる
- **Rationale**: 当時の挙動への忠実さ（legacy）と、リファクタリングで数値が変わらないことの保証（modern）を両立できる
- **Trade-offs**: Optuna が更新されると modern ブロックがずれる可能性があり、そのときは承認を得て更新する（2.5）
- **Follow-up（2026-10-19、Phase 1 の CI）**: SVR の中央値は、macOS arm64（golden の生成環境）と Linux x86_64（CI）とで相対差が最大 4.3e-4 ずれた。値は Python 3.10〜3.14 のどのセルでも同じで、依存の版には左右されない。メンテナーの承認を得て、SVR だけ許容誤差を `rtol=1e-3, atol=1e-3` に緩めた。PLS は `rtol=1e-7, atol=1e-10` のまま

### Decision: golden の生成は「変更前のコード ＋ 名前付きパッチ」に一本化する
- **Context**: 要件 1.8、1.9、1.10、6.3
- **Selected Approach**: 生成スクリプトは常に基準コミット `072d1dfe7ff79b4ebcc411f2b5c13269697e7eac` のコードを `git show` で取り出し、名前付きパッチ（`svr-prediction-shape`、`modern-objective-shape`、`modern-rmse`、`r2-argument-order`、`record-median`）を順に当てて実行する。インストール済みのパッケージは使わない
- **Rationale**: R² 修正後の golden も、変更前のコードにパッチを当てて作れる。こうすると、リファクタリング後のコードを、そのコード自身で作った golden と比べるという循環を避けられる
- **Trade-offs**: 生成スクリプトは Python 3.8 の構文に縛られる

### Decision: アグリゲーションを NumPy に寄せ、pandas を実行時依存から外す
- **Context**: 合成の Simplification。pandas は中央値の計算にしか使っていない
- **Selected Approach**: `np.median` で計算する。pandas はテスト用と example 用の追加依存に移す
- **Rationale**: 中央値は順序統計量なので、pandas でも numpy でも同じ値になる。golden で確かめる
- **Follow-up**: Phase 2 で golden が一致することを必ず確認する

### Decision: 最後の反復の予測中央値を private 属性 `_median_prediction` に残す
- **Context**: 要件 2.8 の比較には中央値が必要。一方で 4.11 は、公開属性として一時状態を残すことを禁じている
- **Selected Approach**: 診断用の private 属性として残し、公開 API には含めない
- **Alternatives**: 予測関数をテストで spy する方法は、内部の呼び出し経路に強く結びつくので採らない

## Risks & Mitigations
- 新しい Optuna の版で SVR の予測がずれる → 要件 2.5 の承認フローで更新する。生成スクリプトの `--env modern` で、ずれの原因が依存にあることを切り分ける
- 2021 年環境が conda-forge から再現できなくなる → ビルド番号まで記録した環境定義を残す。手元の既存環境で動作を確認済み
- リファクタリングで RNG の消費順（P1）や `max_components` の累積最小値（P2）が崩れる → 設計で不変条件として明記し、golden で検出する
- 下限バージョンでのテストは範囲外 → 下限は API が使えることだけを根拠に宣言し、そのことを CHANGELOG に書く

## References
- [scikit-learn 1.8.0 release announcement](https://discuss.scientific-python.org/t/ann-scikit-learn-1-8-0-has-been-released/2196) — 1.8 以降は Python 3.11 以上
- [scikit-learn install guide (1.9)](https://scikit-learn.org/1.9/install.html) — 1.7 は Python 3.10 以上
- [hkaneko1985/ensemble_outlier_sample_detection](https://github.com/hkaneko1985/ensemble_outlier_sample_detection) — 参照実装（予測の連結方法、MAD の式）
- `~/projects/python-template` — pyproject、ruff、mypy、CI、release、dependabot の雛形
