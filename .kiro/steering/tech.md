# Technology Stack

## Architecture

1 モジュールで完結するライブラリ。`EnsembleOutlierSampleDetector`（`sklearn.base.BaseEstimator` を継承）が以下を入れ子のループで実行する。

1. **外側ループ**（最大 `max_iter` 回）: 前回の判定で inlier だったサンプルだけを学習に使う
2. **内側ループ**（`n_estimators` 回）: bootstrap → ゼロ分散特徴量を除去 → `X` と `y` を標準化 → Optuna でハイパーパラメータを探索（目的関数は `cross_val_predict` による OOF スコア）→ 最良パラメータで再学習 → **元の全サンプル**を予測
3. **判定**: `|y - median(予測)| > 3 * 1.4826 * MAD` を満たすサンプルを outlier とする。前回と mask が同じになったら収束とみなす

Optuna の目的関数 `Objective` は、検出器への後方参照（`elo`）を通じて、`fit` が bootstrap ごとに書き込む状態（`X_bootstrapped`、`y_bootstrapped`、`scaler_y`、`metric`、`cv`、`n_jobs`）を読む。そのため、`fit` 側で状態の持ち方を変えるときは `Objective` も合わせて変える必要がある。

## Core Technologies

- **Language**: Python。現状の classifiers は 3.6–3.9。移行後の目標は `requires-python >= 3.10`
- **Framework**: scikit-learn の estimator 規約（`BaseEstimator`、`check_X_y`、`check_is_fitted`）
- **Runtime 依存**: numpy、pandas、scikit-learn、optuna、tqdm。`requirements.txt` で下限を固定しているのは `optuna>=0.19.0` だけ

## Key Libraries

- **Optuna**: `TPESampler(seed=...)` でサブモデルごとにハイパーパラメータを探索する。ログは `optuna.logging.disable_default_handler()` で抑制している
- **scikit-learn**: `PLSRegression(scale=False)`、`SVR(kernel="rbf")`、`resample`、`VarianceThreshold`、`StandardScaler`
- **tqdm**: `'ipykernel' in sys.modules` のときは `tqdm.notebook` に切り替える

## Development Standards

### 現状
テスト、linter、formatter、型チェック、CI はどれもない。`example/example.ipynb` を手で実行して動作を確かめている。

### 移行後の目標（`~/projects/python-template` に準拠）
- **Lint / Format**: `ruff.toml` で `line-length = 79`、pydocstyle は `convention = "numpy"`
- **Type Safety**: `mypy.ini`（`ignore_missing_imports = True`）。公開 API にはすべて型ヒントを付ける
- **Docstring**: NumPy style。Parameters / Returns / Attributes / Raises を書く
- **Testing**: pytest（`tests/`）。固定 `random_state` での検出結果を golden ファイルと照合し、リファクタリング前後で数値結果が変わらないことを保証する
- **CI**: GitHub Actions でテスト・lint・型チェックを回す。release ワークフローと dependabot も入れる

## Development Environment

### Common Commands
```bash
# 現状
pip install -r requirements.txt && pip install -e .
python setup.py sdist bdist_wheel     # PyPI 用ビルド

# 移行後（予定）
pip install -e ".[test]"
pytest
ruff check . && ruff format --check .
mypy src
python -m build
```

## Key Technical Decisions

- **単一 RNG による再現性**: `check_random_state(random_state)` で作った RNG 1 つから、bootstrap と TPE sampler の seed を毎回 `randint(2**32)` で引いている。seed を引く順序を変えると結果が変わるので、リファクタリングでは順序を保つ
- **スコアは元スケールで計算**: `y` を標準化して学習し、OOF 予測と `y` を `scaler_y.inverse_transform` で元のスケールに戻してから metric を計算する
- **Optuna の試行回数**: `min(100, max_components)` 回。PLS の場合、`max_components` は bootstrap データの行列ランク以下に抑える
- **古い API への依存**: `np.bool`、`mean_squared_error(squared=False)`、`trial.suggest_loguniform` を使っている。置き換えるときは数値結果が変わらないことを golden テストで確かめる
- **ライセンス**: Apache License 2.0。python-template は MIT なので、テンプレートからライセンス表記を持ち込まない

---
_Document standards and patterns, not every dependency_
_updated_at: 2026-10-19_
