# Project Structure

## Organization Philosophy

src-layout の小さなライブラリ。公開モジュールは 1 つで、そこから責務ごとの private モジュールに処理を切り出している。`__init__.py` は公開 API の再エクスポートとメタデータの置き場にとどめる。検出結果は golden 資産で固定し、テストとは別のディレクトリで管理する。

## Directory Patterns

### パッケージ本体
**Location**: `src/ensemble_outlier_sample_detection/`
**Purpose**: 公開 API とその実装
**Example**:
- `ensemble_outlier_sample_detection.py` が公開クラス `EnsembleOutlierSampleDetector` を持つ
- `_submodels.py`（探索空間、モデルの生成、目的関数）と `_runtime.py`（ログ抑制、進捗バー）は、副作用や関心ごとを分けるための private モジュール
- `__init__.py` は `from .ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector` で明示的に再エクスポートし、`__all__ = ["EnsembleOutlierSampleDetector"]` で公開範囲を限定する
- `_version.py` は setuptools-scm が生成する（git 管理外）

### テスト
**Location**: `tests/`
**Purpose**: pytest のテスト
**Example**:
- 対象ごとに `test_<対象>.py` を置く（`test_estimator.py`、`test_submodels.py`、`test_runtime.py`、`test_golden.py`、`test_import.py`）
- 共有の fixture（参照データ、golden case の読み込み、golden が書き換えられていないかの確認）は `conftest.py` に置く
- golden の照合ロジックは補助モジュール `_golden_cases.py` にまとめる。tests/ には `__init__.py` を置かず、pytest 既定の prepend import mode でトップレベルの名前として import する

### golden 資産
**Location**: `tests/golden/`
**Purpose**: 変更前の検出結果を固定する資産と、その生成手順
**Example**:
- `cases/{method}-{metric}.json`: case ファイル（schema_version 1、`legacy` / `modern` の 2 ブロック）。手で編集せず、生成スクリプトで書き出す
- `generate_golden.py`: 生成スクリプト。2021 年環境でも動かすため、Python 3.8 互換の構文で書く
- `environment-legacy.yml` / `requirements-modern.txt`: 2 つの生成環境の定義
- `README.md`: 生成手順、更新ルール、更新履歴（承認の記録）

### サンプル
**Location**: `example/`
**Purpose**: notebook による使用例と、それに使う参照データ
**Example**:
- `example_chache.csv` の先頭は UTF-8 BOM と、原データの出典（MIT License）を書いた `#` コメント行。読み込みには `comment="#"` が要る。golden の参照データも兼ねるので、内容を変えない（case ファイルの `data.sha256` で検出する）
- notebook は ruff の対象外。wheel にも sdist にも入れない（CSV だけ sdist に入れる）

### 自動化
**Location**: `.github/`
**Purpose**: CI、リリース、依存の更新
**Example**: `workflows/CI.yml`（test / build / lint）、`workflows/release.yml`（タグで公開）、`dependabot.yml`、`release.yml`（リリースノートのカテゴリ）

### 利用者・メンテナー向けの文書
**Location**: リポジトリのルート
**Purpose**: README（英語。PyPI にも表示される）、`CHANGELOG.md`（英語。Keep a Changelog 形式）、`HOW_TO_RELEASE.md`（日本語。メンテナー向けの公開手順）

## Naming Conventions

- **Files / Modules**: snake_case。公開モジュールにはパッケージと同じ名前を付け、private モジュールは先頭に `_` を付ける
- **Classes**: PascalCase（`EnsembleOutlierSampleDetector`、`MetricSpec`）
- **Fitted attributes**: sklearn と同じく、公開する属性の末尾に `_` を付ける（`outlier_support_`、`inlier_support_`、`n_iter_`、`n_features_in_`、`feature_names_in_`）。テスト用の診断値は、先頭に `_` を付けた private 属性にする（`_median_prediction`）
- **Private helpers**: 先頭に `_` を付ける（`_check_positive_int`、`_fit_predict_submodel`）

## Import Organization

```python
# 標準ライブラリ → サードパーティ → パッケージ内（ruff の I ルールで並べる）
import numbers
import warnings

import numpy as np
import optuna
from sklearn.base import BaseEstimator

# private モジュールは名前ごと import し、呼び出すときに属性として参照する
from . import _runtime, _submodels
```

private モジュールの関数は `from ._runtime import make_progress_bar` の形で取り込まない。テストが検出器モジュールの `_runtime.make_progress_bar` などを差し替えて検証しているため、属性として参照する形を保つ（取り込む形に変えると、テストが明示的なメッセージで失敗する）。

## Packaging Metadata

- ビルド設定とメタデータは `pyproject.toml` に一元化している（setuptools + setuptools-scm）。`setup.py` と `requirements.txt` はない
- バージョンは git タグから決まり、`version_file` で `_version.py` に書き出される。`__init__.py` は `_version.py` がなければ `__version__ = "unknown"` にする
- `__init__.py` に残すメタデータは `__license__`、`__author__`、`__copyright__` だけ
- `MANIFEST.in` で、sdist から notebook と開発用ディレクトリ（`.kiro`、`.claude`、`.github`）を除く。テストと参照 CSV は含める

## Code Organization Principles

- 公開 API は `__all__` で明示し、それ以外は内部実装として扱う
- コンストラクタは引数をそのまま保存するだけにする。検証と正規化は `fit` の冒頭でまとめて行い、途中の状態は `fit` の局所変数に置く（`fit` の後に公開属性として残すのは、学習結果だけ）
- グローバルな状態を変える処理（Optuna のログ設定、進捗バー）は、context manager で囲み、例外のときも元に戻す
- 探索や評価の部品は、検出器の状態を読まず、必要な値を引数で受け取る
- 計算の順序や乱数の消費を変える変更は、golden テストで検出結果が変わらないことを確かめる。意図して結果を変えるときは、`tests/golden/README.md` の更新ルールに従う

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
_updated_at: 2026-10-20（1.0.0 の実装完了にあわせて、移行期の記述を今の状態に更新）_
