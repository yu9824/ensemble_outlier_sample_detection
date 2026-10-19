# golden（変更前の検出結果）

このディレクトリには、本 spec でコードを変える前の検出結果を固定した golden と、その生成手順を置いている。リファクタリングや依存の更新で検出結果が変わっていないかを、テストがこの golden と照合して確かめる。

| ファイル | 内容 |
|----------|------|
| `cases/{method}-{metric}.json` | 8 構成ぶんの golden（手で編集しない） |
| `generate_golden.py` | golden の生成スクリプト（Python 3.8 互換） |
| `environment-legacy.yml` | 2021 年環境の conda 定義（ビルド番号まで固定） |
| `requirements-modern.txt` | 最新環境の pip 固定 |

## golden の中身

基準コミットは `072d1dfe7ff79b4ebcc411f2b5c13269697e7eac`（2021-07-15）。生成スクリプトはこのコミットのソースを `git show` で取り出し、後述の名前付きパッチを当ててから `eosd_baseline` という別名で import する。インストール済みのパッケージや作業ツリーのコードは使わない。

case ファイル 1 つにつき、ブロックが 2 つある。

| ブロック | 生成環境 | テストで比べるもの |
|----------|----------|--------------------|
| `legacy` | 2021 年環境（Python 3.8、NumPy 1.20、scikit-learn 0.24、pandas 1.2、Optuna 2.8） | `outlier_support`、`inlier_support`、`n_iter` を完全一致で比べる。`median_prediction` は参考として残すだけで、比べない |
| `modern` | 最新環境（Python 3.12、NumPy 2.5、scikit-learn 1.9、pandas 3.0、Optuna 5.0） | mask と `n_iter` を完全一致で比べる。`median_prediction` は、PLS を `rtol=1e-7`、`atol=1e-10` で、SVR を `rtol=1e-3`、`atol=1e-3` で比べる |

`n_iter` は実行した反復の回数で、変更前のコードの `n_iter_finished + 1` にあたる。`median_prediction` は、最後の反復におけるアンサンブル予測の中央値（サンプルごと）である。各ブロックの `provenance` には、基準コミット、当てたパッチ、Python と依存の版、生成日時、生成した理由が入る。入力データは `example/example_chache.csv`（100 サンプル、10 特徴量）で、ファイルの SHA-256 を `data.sha256` に記録している。

### 名前付きパッチ

変更前のコードは、そのままでは SVR が動かない。最新環境では、削除された API を使っている rmse の構成も動かない。そこで、計算を変えない最小限の修正だけを、完全一致する文字列を 1 回だけ置換する形で当てている。置換箇所が 1 回で一致しなければ、スクリプトは中断する。

| パッチ | 当てる対象 | 理由 |
|--------|------------|------|
| `svr-prediction-shape` | 両方の環境の全構成 | SVR の予測は 1 次元なので、`inverse_transform` の前に列ベクトルに直す（要件 1.8）。PLS の結果を変えないことは、2021 年環境で毎回確かめる（要件 1.9） |
| `modern-objective-shape` | 最新環境の全構成 | 目的関数の中の `inverse_transform` に渡す SVR の OOF 予測も、同じく列ベクトルに直す。新しい scikit-learn は 1 次元の入力を受け付けない。PLS の予測はもともと列ベクトルなので、PLS には影響しない（要件 1.10） |
| `modern-rmse` | 最新環境の全構成 | `mean_squared_error(..., squared=False)` は削除されたので、`np.sqrt(mean_squared_error(...))` に置き換える（要件 1.10） |
| `record-median` | 両方の環境の全構成 | 予測行列を作った直後に、中央値を `_median_prediction` に記録する。記録するだけで、計算は変えない（要件 1.10） |
| `r2-argument-order` | `--patch r2-argument-order` を付けたときの r2 構成だけ | R² の引数順を定義どおりに直す（要件 6.3）。初回の生成では当てていない。Phase 3 で作り直してからは、`pls-r2` と `svr-r2` の両方のブロックをこのパッチ込みで生成している（パッケージの R² の修正に合わせるため）。r2 構成を作り直すときは、必ずこのパッチを付ける |

### 8 つの構成

手法（`pls`、`svr`）と評価指標（`r2`、`mse`、`mae`、`rmse`）のすべての組み合わせを、`pls-r2` のような名前で持つ。共通のパラメータは `random_state=334`、`max_iter=30`、`cv=5`、`max_components=30`、`n_jobs=1`、`progress_bar=False`。`n_estimators` だけは CI の時間予算に合わせて、PLS を 30、SVR を 5 にしている。

| 構成 | 外れ値の数 | `n_iter` |
|------|-----------:|---------:|
| `pls-r2` / `pls-mse` / `pls-mae` / `pls-rmse` | 15 | 7 |
| `svr-r2` / `svr-mse` / `svr-mae` / `svr-rmse` | 0 | 1 |

初回の生成でも Phase 3 の再生成でも、どの構成も共通のパラメータのまま `max_iter` 内に収束した。そのため `generate_golden.py` の `PARAM_OVERRIDES` は空のままにしている（要件 1.6 の調整はなし）。R² の修正の前は、`pls-r2` の `n_iter` は 8 だった（外れ値の 15 件は修正の前後で同じ）。

## 環境の作り方

2021 年環境は osx-arm64 専用である。`environment-legacy.yml` はビルド番号まで固定しているので、ほかのプラットフォームでは解決できない。

```sh
mamba env create -f tests/golden/environment-legacy.yml   # 環境名は eosd-golden-legacy
```

最新環境は conda-forge の Python を入れてから、pip で固定した版を入れる。開発ツールは golden 用の固定とは分けて、版を固定せずに入れる。

```sh
mamba create -n eosd-modern -c conda-forge --override-channels python=3.12.15 pip
mamba activate eosd-modern
python -m pip install -r tests/golden/requirements-modern.txt
python -m pip install pytest build ruff mypy
```

どちらの環境でも、生成スクリプトが最初に Python と主要な依存（numpy、scipy、scikit-learn、pandas、optuna、tqdm）の版を確かめ、固定値と違えば何もせずに終了コード 1 で終わる。ただし最新環境では、Python は 3.10 以上かどうかだけを見て、パッチ版までは確かめない。期待する版は `generate_golden.py` の定数にも書いてあるので、固定ファイルを変えたときは定数も合わせて直す。生成には git と、基準コミットを含む clone が要る。

## 生成コマンド

リポジトリのルートで、対応する環境の Python から実行する。1 回の実行は 1〜2 分かかる。

```sh
mamba activate eosd-golden-legacy
python tests/golden/generate_golden.py --env legacy   # legacy ブロックを書く

mamba activate eosd-modern
python tests/golden/generate_golden.py --env modern   # modern ブロックを書く
```

1 回の実行で書くのは `--env` に対応するブロックだけで、もう一方のブロックはそのまま残す。結果、当てたパッチ、基準コミットが既存のブロックと同じなら、ファイルには触れない。違う場合は、`--force` と `--reason` を両方付けない限り、どのファイルも書かずに終了コード 1 で終わる。1 つでも収束しない構成があったときや、PLS のパッチ中立性の確認が通らなかったときも、何も書かない。

よく使うオプションは次の 4 つ。

```sh
# 一部の構成だけを生成する
python tests/golden/generate_golden.py --env modern --cases pls-r2 svr-r2

# 承認を得た更新で、既存のブロックを上書きする（理由は provenance に残る）
python tests/golden/generate_golden.py --env modern --force --reason "TEXT"

# Phase 3: R² の修正を当てて、r2 構成だけを両方の環境で作り直す
python tests/golden/generate_golden.py --env legacy --cases pls-r2 svr-r2 \
    --patch r2-argument-order --force --reason "R² argument order fix (Req 6.3)"
python tests/golden/generate_golden.py --env modern --cases pls-r2 svr-r2 \
    --patch r2-argument-order --force --reason "R² argument order fix (Req 6.3)"

# 診断用: golden の写しに対して実行し、tests/golden/cases は変えない
cp -R tests/golden/cases /path/to/scratch/cases
python tests/golden/generate_golden.py --env modern --cases-dir /path/to/scratch/cases
```

`--cases-dir` に golden の写しを渡すと、結果が同じなら `unchanged` と表示し、違えば一致しないフィールドとサンプルの index を表示して終了コード 1 で終わる。どちらの場合も写しは書き換わらない。

実行中に stderr へ出る警告の件数表示は、変更前のコードがもともと出しているもので、想定どおりである。2021 年環境では `np.bool` の DeprecationWarning が、SVR では DataConversionWarning が、最新環境の SVR では `suggest_loguniform` の FutureWarning が出る。

## 更新ルール

case ファイルは手で編集しない。書き換えるのは生成スクリプトだけで、テストは読むだけにする。

テストが golden と一致しなくなったら、まず原因がどちらにあるかを切り分ける。

1. 本パッケージのコードを変えたことが原因なら、golden は更新せずにコードを直す（要件 2.4）。例外は、Phase 3 の R² の修正（要件 6.3）のように、承認を得て検出結果を意図的に変える場合だけで、そのときは基準コミットに同じ修正を名前付きパッチとして当てて作り直す。
2. 依存の更新が原因だと疑われるなら、変更前のコードを新しい依存で動かして、同じずれが出るかを確かめる。生成スクリプトは常に基準コミットのコードを動かすので、新しい依存の環境を作り、`requirements-modern.txt` と `generate_golden.py` の `MODERN_VERSIONS` をその版に合わせてから、上の診断用コマンド（`--env modern --cases-dir`）を実行すればよい。
3. 診断で `unchanged` と出たなら、ずれの原因は依存ではなくコードにある。1 に戻る。
4. 診断でテストと同じずれが出たときだけ、メンテナーの承認を得て `--force --reason` で更新し、下の更新履歴に 1 行足す。原因の切り分けの結果、依存の版、承認者を必ず書く（要件 2.5）。

params や入力データを変えた場合も、`--force --reason` がなければ書き込まない。書き込むと、もう一方の環境のブロックが古くなったことを警告するので、その環境でも `--cases NAME --force --reason TEXT` を付けて作り直す。

最新環境で `legacy` ブロック（mask と `n_iter`）との照合がずれた場合も、同じように扱う。2021 年環境は固定なので、作り直しても `unchanged` になり、ずれは解消しない。CI（Linux x86_64、Python ごとに依存の版が違う）でだけ一致しないセルが出た場合も、golden は更新しない。作業を止め、設計の Open Questions にある選択肢（許容誤差の見直し、環境ごとの variant、明示的な skip）をメンテナーに示して判断を仰ぐ。

## 知っておくこと

SVR の中央値は、golden を生成した macOS arm64 と CI の Linux x86_64 とで、相対差が最大で約 4.3e-4、絶対差が最大で約 7.9e-4 ずれる（2026-10-19 の CI で確認）。ずれの値は Python 3.10〜3.14 のどのセルでも同じで、依存の版（NumPy 2.2〜2.5、scikit-learn 1.7〜1.9、pandas 2.3〜3.0）には左右されない。mask と `n_iter` は一致し、PLS の中央値も厳しい許容誤差のまま一致する。そこでメンテナーの承認を得て、SVR の中央値だけ許容誤差を `rtol=1e-3`、`atol=1e-3` に緩めた。Optuna の版の違いで選ばれるハイパーパラメータが変わると差は 0.59〜0.87 になるので、この許容誤差でもその種の変化は検出できる。


SVR はどの構成でも外れ値を 1 件も検出せず、1 反復で収束する。mask がすべて `false` なので、SVR の mask の照合で見つかる変化は少ない。SVR の変化を実際に捕まえるのは、`modern` ブロックの中央値の照合である。

2021 年環境と最新環境の中央値は、PLS では約 1.8e-15 しか違わないが、SVR では最大で約 0.87 違う（`svr-mae` が 0.87、`svr-r2`、`svr-mse`、`svr-rmse` が 0.59。`svr-r2` は R² の修正の前は 1.09 だった）。Optuna 2.8 と 5.0 の TPE の違いで、選ばれるハイパーパラメータが変わるためと考えられる（PLS の探索は試行が少なくランダムサンプリングだけで終わるので、差が出ない）。mask と `n_iter` は、8 構成すべてで両方の環境が一致している。`legacy` の中央値を比べないのは、このためである。

NumPy 2 では `np.bool` が `np.bool_` の別名として戻ったので、最新環境で `np.bool` を直すパッチは要らない。

2021 年環境では、警告を表示する設定（`python -W default` や pytest）で `import optuna` を実行すると、SQLAlchemy の MovedIn20Warning が出る。推移的依存の SQLAlchemy を参照環境の版（2.0）に合わせたためである。golden はインメモリのストレージで生成するので、計算には影響しない。

golden は macOS arm64 で生成した。CI は Linux x86_64 で動くので、BLAS などの違いで中央値がずれる可能性がある。その場合の扱いは、上の更新ルールの最後の段落のとおり。

## 更新履歴

依存の版は、Python、NumPy、SciPy、scikit-learn、pandas、Optuna、tqdm の順。生成したプラットフォームは、どちらも macOS arm64 である。

| 日付 | case | ブロック | 理由 | 原因の切り分け | 依存の版 | 承認者 |
|------|------|----------|------|----------------|----------|--------|
| 2026-10-19 | 全 8 構成 | legacy | initial baseline | 該当なし（初回生成） | 3.8.20 / 1.20.1 / 1.6.2 / 0.24.1 / 1.2.4 / 2.8.0 / 4.59.0 | yu9824（spec の承認による） |
| 2026-10-19 | 全 8 構成 | modern | initial baseline | 該当なし（初回生成） | 3.12.15 / 2.5.4 / 1.18.1 / 1.9.1 / 3.0.6 / 5.0.0 / 4.70.1 | yu9824（spec の承認による） |
| 2026-10-19 | `pls-r2`、`svr-r2` | legacy | R² の引数順の修正（要件 6.3）。`r2-argument-order` を当てて再生成。`pls-r2` は `n_iter` が 8 から 7 に、両構成で中央値が変わった。mask は変わらない | コードの意図的な修正（R² を実測値・予測値の順に） | 3.8.20 / 1.20.1 / 1.6.2 / 0.24.1 / 1.2.4 / 2.8.0 / 4.59.0 | yu9824（spec 6.3 の承認による） |
| 2026-10-19 | `pls-r2`、`svr-r2` | modern | R² の引数順の修正（要件 6.3）。`r2-argument-order` を当てて再生成。`pls-r2` は `n_iter` が 8 から 7 に、両構成で中央値が変わった。mask は変わらない | コードの意図的な修正（R² を実測値・予測値の順に） | 3.12.15 / 2.5.4 / 1.18.1 / 1.9.1 / 3.0.6 / 5.0.0 / 4.70.1 | yu9824（spec 6.3 の承認による） |
