# Technology Stack

## Architecture

公開モジュール 1 つと private モジュール 2 つからなる小さなライブラリ。依存の向きは `_runtime` / `_submodels` → `ensemble_outlier_sample_detection`（検出器）→ `__init__` の一方向で、private モジュール同士は import しない。

- **`EnsembleOutlierSampleDetector`**（`sklearn.base.BaseEstimator` を継承）: 入力検証、反復ループ、判定を受け持つ
  1. **外側ループ**（最大 `max_iter` 回）: 前回の判定で inlier だったサンプルだけを学習に使う
  2. **内側ループ**（`n_estimators` 回）: bootstrap → ゼロ分散の特徴量を除く → `X` と `y` を標準化 → Optuna でハイパーパラメータを探索（目的関数は `cross_val_predict` の OOF スコア）→ 最良のパラメータで学習し直す → **元の全サンプル**を予測して、予測行列の 1 列に書く
  3. **判定**: `|y - median(予測)| > 3 * 1.4826 * MAD` を満たすサンプルを outlier とする。mask が前回と同じか、inlier が 0 件になったら収束とみなす
- **`_submodels`**: 手法と評価指標の正規化（未対応の値は `ValueError`）、探索空間、モデルの生成、試行回数、目的関数（必要な値を引数で受け取る closure で、検出器の状態は読まない）
- **`_runtime`**: Optuna のログ抑制（context manager）と進捗バーの生成

## Core Technologies

- **Language**: Python ≥ 3.10。CI では 3.10〜3.14 でテストする
- **Framework**: scikit-learn の estimator 規約（`BaseEstimator`、`validate_data`、`check_is_fitted`、`ConvergenceWarning`）
- **Runtime 依存**: `numpy>=1.23`、`scikit-learn>=1.6`、`optuna>=3.0`、`tqdm>=4.40`。pandas は実行時依存ではなく、テスト用と example 用の追加依存にだけ入っている（DataFrame の入力は scikit-learn 側で扱える）。下限は API が使えることを根拠に決めたもので、下限の版でのテストはしていない

## Key Libraries

- **Optuna**: サブモデルごとに `TPESampler(seed=...)` で探索する。fit の間だけ verbosity を WARNING にし、`finally` で元の値に戻す。default handler には触らない
- **scikit-learn**: `PLSRegression(scale=False)`、`SVR(kernel="rbf")`、`resample`、`VarianceThreshold`、`StandardScaler`、`root_mean_squared_error`
- **tqdm**: `tqdm.auto` が notebook かどうかを判定する。`progress_bar=False` のときは `disable=True` にし、同じコードの経路で無効にする。無効にしても `tqdm.write` は出力されるので使わない

## Development Standards

### Type Safety
- `mypy.ini` で `files = src`、`disallow_untyped_defs = True`、`ignore_missing_imports = True`、`python_version = 3.10`。src の関数とメソッドには、すべて型ヒントを付ける
- 型情報を同梱していることを示す `py.typed` を配布物に含める

### Code Quality
- `ruff.toml`: `line-length = 79`、`extend-select = ["D", "I"]`、pydocstyle は `convention = "numpy"`、`tests/**` には D ルールを適用しない
- ruff は Markdown の中の Python コード片も整形する。README や CHANGELOG のコード例も `ruff format` の対象になる。spec 文書（`.kiro`）とエージェント用の設定（`.claude`）、`example/` は対象から外している
- 公開 API の docstring は NumPy スタイルで書き、取りうる値と既定値、戻り値、学習後の属性、Raises / Warns を記す

### Testing
- pytest（`tests/`）。`pyproject.toml` の `filterwarnings` で、自パッケージが原因の `DeprecationWarning` / `FutureWarning` をエラーにする
- **golden テストが安全網**。8 構成（pls / svr × r2 / mse / mae / rmse）について、変更前のコードの結果と照合する
  - `legacy` ブロック（2021 年環境）: mask と `n_iter` を完全一致で比べる
  - `modern` ブロック（最新環境）: mask と `n_iter` を完全一致で比べ、最後の反復の予測中央値 `_median_prediction` を許容誤差付きで比べる。許容誤差は PLS が `rtol=1e-7, atol=1e-10`、SVR が `rtol=atol=1e-3`（macOS と Linux とで SVR の予測が最大 4e-4 ずれるため）
- 反復回数を前提にしたテストの seed は実測で選んでいる（`SMALL` / `FAST` は 12、`SEED_3_ITERATIONS` は 8）。既定の metric は r2 なので、探索や評価の挙動を変えると seed を測り直す必要がある。前提が崩れれば、assert で明示的に失敗する
- 速さの目安は、ローカルで全体 30〜60 秒、CI でジョブあたり 10 分以内（`timeout-minutes: 10`）

### CI / Release
- `CI.yml`: 変更を検知したうえで、test（3.10〜3.14、`fail-fast: false`）、build（wheel に `example/` と `tests/` が入らず、`py.typed` が入ること）、lint（ruff check / format、mypy）を回す。CI は `pull_request` と main への push で起動する
- `release.yml`: `v*.*.*` のタグで起動し、sigstore で署名した GitHub Release を作る。安定版と `-rc.N` は PyPI に、すべてのタグは TestPyPI に、trusted publishing（OIDC）で公開する。タグ名はシェルに `${{ }}` で埋め込まず、引用符付きの `"$GITHUB_REF_NAME"` で参照する
- dependabot は pip と GitHub Actions の更新を週 1 回提案する

## Development Environment

### Required Tools
- golden の生成には、conda 環境を 2 つ使う（ローカルだけで使い、CI では生成しない）
  - `eosd-golden-legacy`: `tests/golden/environment-legacy.yml` から作る。Python 3.8、osx-arm64 専用
  - `eosd-modern`: Python 3.12 に `tests/golden/requirements-modern.txt` を入れたもの
- 日々の開発とテストも `eosd-modern` で行える（pytest、build、ruff、mypy が入っている）

### Common Commands
```bash
pip install -e ".[test,dev]"     # 開発用インストール
pytest                           # テスト（golden を含む）
ruff check . && ruff format --check .
mypy                             # 対象は mypy.ini の files = src
python -m build                  # sdist と wheel
# golden の再生成と診断（手順と更新ルールは tests/golden/README.md）
python tests/golden/generate_golden.py --env {legacy|modern} [--cases ...] [--cases-dir DIR]
```

## Key Technical Decisions

- **単一の RNG による再現性**: `check_random_state(random_state)` で作った 1 つの RNG から、bootstrap ごとに resample → TPE の順で `randint(2**32)` を 2 回引く。引く順序を変えると結果が変わる（golden で検出できる）
- **PLS の `max_components` は累積の最小値**: bootstrap ごとに、標準化した bootstrap の行列ランクで上限を下げ、bootstrap と反復をまたいで持ち越す。fit の局所変数で持ち、コンストラクタ引数は書き換えない
- **スコアは元のスケールで計算**: `y` を標準化して学習し、OOF 予測と `y` を `scaler_y.inverse_transform` で元のスケールに戻してから、`score(y_true, y_pred)` の順で評価する（0.x では r2 の引数が逆だった。1.0.0 で修正）
- **サブモデルには 1 次元の `y` を渡す**: 予測は常に `(n, 1)` にそろえてから逆変換する。2 次元で渡していた 0.x と数値は同じで、SVR の形の不具合と `DataConversionWarning` がなくなる
- **golden は基準コミットのコードから作る**: 生成スクリプトは基準コミット `072d1df` のソースを `git show` で取り出し、名前付きのパッチ（文字列を 1 回だけ置換するもの）を当てて動かす。インストール済みのパッケージは使わないので、リファクタリング後のコードを自分自身の出力で検証する循環が起きない
- **ライセンス**: Apache License 2.0。python-template は MIT なので、テンプレートからライセンス表記を持ち込まない

---
_Document standards and patterns, not every dependency_
_updated_at: 2026-10-20（1.0.0 の実装完了にあわせて、移行期の記述を今の状態に更新）_
