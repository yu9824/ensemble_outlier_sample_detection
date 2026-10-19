# Project Structure

## Organization Philosophy

ロジックを 1 つのモジュールに集約した小さなライブラリ。パッケージの `__init__.py` は公開 API の再エクスポートとメタデータの置き場、実装モジュールはアルゴリズム本体、と役割を分けている。現在は flat layout だが、python-template に合わせて src-layout へ移行する予定。

## Directory Patterns

### パッケージ本体
**Location**: `ensemble_outlier_sample_detection/`（移行後は `src/ensemble_outlier_sample_detection/`）
**Purpose**: 公開 API とその実装
**Example**: `__init__.py` が `from .ensemble_outlier_sample_detection import *` で再エクスポートし、`__all__ = ["EnsembleOutlierSampleDetector"]` で公開範囲を限定している

### サンプルと手動検証
**Location**: `example/`
**Purpose**: notebook による使用例と、それに使うデータ
**Example**: `example_chache.csv` の先頭は UTF-8 BOM と、原データの出典を書いた `#` コメント行。読み込みには `comment="#"` が必要。`.gitattributes` で `linguist-vendored` に指定しており、パッケージにも含めない

### テスト（移行後）
**Location**: `tests/`
**Purpose**: pytest のテストと golden ファイル
**Example**: python-template と同じく `test_<対象>.py` という名前にする。golden の基準値は旧依存環境で現行コードを実行して作る

## Naming Conventions

- **Files / Modules**: snake_case。実装モジュールにはパッケージと同じ名前を付けている
- **Classes**: PascalCase（`EnsembleOutlierSampleDetector`、`Objective`）
- **Fitted attributes**: sklearn と同じく末尾に `_` を付ける（`outlier_support_`、`inlier_support_`、`n_iter_`）。例外として `n_iter_finished` には `_` が付いていない
- **Private helpers**: 先頭に `_` を付ける（`_extract`）

## Import Organization

```python
# 標準ライブラリ → サードパーティの順に絶対 import する
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import optuna

# パッケージ内はすべて相対 import
from .ensemble_outlier_sample_detection import *
```

## Packaging Metadata

- **現状**: `__init__.py` の `__version__` などの単一行文字列リテラルを、`setup.py` が正規表現で抜き出して使う。`setup.py` に残っている `py2app` の分岐は使っていない
- **移行後**: `pyproject.toml` に一本化し、バージョンは setuptools-scm で git タグから生成する（`_version.py` に書き出す）。`requirements.txt` と `setup.py` は廃止する

## Code Organization Principles

- 公開 API は `__all__` で明示し、それ以外は内部実装として扱う
- `Objective` のような内部クラスは検出器の状態を直接読む密結合な作りになっている。分離するときは、bootstrap ごとの状態を引数で明示的に渡す形にする
- 移行後は、`fit` でコンストラクタ引数を書き換えず、学習で決まる値は `*_` 属性に保存する（sklearn 規約）

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
_updated_at: 2026-10-19_
