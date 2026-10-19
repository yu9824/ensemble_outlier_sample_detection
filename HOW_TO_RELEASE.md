# リリース手順

バージョンタグを push すると、`.github/workflows/release.yml` がビルドから公開までを行う。版は setuptools-scm がタグから決めるので、`pyproject.toml` やソースに版を書く必要はない。

## 事前準備（初回だけ）

### GitHub の environment を作る

リポジトリの Settings → Environments で、`pypi` と `testpypi` の 2 つを作る。release.yml の公開ジョブは、この名前の environment の中で動く。存在しない environment はジョブの初回実行時に GitHub が自動で作るが、保護ルールを付けるなら先に作っておく。

保護ルールは任意である。付ける場合は次の点に気をつける。

- `pypi` に Required reviewers を付けると、PyPI に公開する前に承認を求められる
- Deployment branches and tags で実行できる ref を絞るなら、Selected branches and tags を選び、Ref type を Tag にして `v*` を加える。公開ジョブはタグの ref で動くので、ブランチだけを許可すると公開できない

設定画面の詳細は [GitHub のドキュメント](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments) を参照。

### PyPI と TestPyPI に trusted publisher を登録する

公開には trusted publishing（GitHub Actions の OIDC トークンによる認証）を使う。API トークンは発行せず、リポジトリの secrets にも置かない。PyPI と TestPyPI のそれぞれで、GitHub Actions の publisher として次の値を登録する。

| 入力欄 | PyPI | TestPyPI |
| --- | --- | --- |
| PyPI Project Name | `ensemble-outlier-sample-detection` | `ensemble-outlier-sample-detection` |
| Owner | `yu9824` | `yu9824` |
| Repository name | `ensemble_outlier_sample_detection` | `ensemble_outlier_sample_detection` |
| Workflow name | `release.yml` | `release.yml` |
| Environment name | `pypi` | `testpypi` |

Environment name は省略できる欄だが、必ず入れる。省略すると、release.yml の中で `id-token: write` を持つ別のジョブ（sigstore の署名に使う `github-release` や、TestPyPI に公開するジョブ）の OIDC トークンでも PyPI に公開できてしまう。

登録する画面は、プロジェクトがすでにあるかどうかで変わる。

- プロジェクトがある場合は、そのプロジェクトの管理画面の Publishing から追加する。この画面では PyPI Project Name を入れない
- まだない場合は、アカウントの Publishing の画面から pending publisher として追加する。最初の公開のときにプロジェクトが作られる

このプロジェクトは PyPI と TestPyPI の両方に 0.1.0 がすでにある（2026-10-19 時点）。そのため、どちらもプロジェクトの Publishing から登録する。TestPyPI のプロジェクトが自分の TestPyPI アカウントの持ち物であることも、先に確かめておく。

画面の操作は [PyPI の trusted publishing のドキュメント](https://docs.pypi.org/trusted-publishers/) を参照。

## リリースの流れ

1. リリースする変更を main にマージし、main の CI が通っていることを確かめる
2. `CHANGELOG.md` に、リリースする版の節があることを確かめる
3. main の最新のコミットにタグを付け、そのタグだけを push する

```sh
git switch main
git pull --ff-only
git tag v1.0.0
git push origin v1.0.0
```

`git push --tags` は使わない。手元にある別のタグまで push してしまう。

push すると、GitHub の Actions に `Publish Python 🐍 distribution 📦 (v1.0.0)` という実行が現れる。`build` ジョブのあとに、残りの 3 つのジョブが並んで動く。

| ジョブ | 終わったあとに確かめること |
| --- | --- |
| `build` | ログに出る sdist と wheel のファイル名の版が、タグと一致している |
| `github-release` | GitHub の Releases にタグと同じ名前のリリースができ、sdist、wheel、sigstore の署名ファイルが添付されている |
| `publish-to-pypi` | <https://pypi.org/project/ensemble-outlier-sample-detection/> に新しい版が出ている |
| `publish-to-testpypi` | <https://test.pypi.org/project/ensemble-outlier-sample-detection/> に新しい版が出ている |

リリースノートは、マージした PR から自動で作られる。PR のラベル `enhancement` と `bug` で分類されるので（`.github/release.yml`）、マージする前にラベルを付けておく。

Actions の画面からワークフローを手動で実行することもできる。ブランチを選んで実行すると `build` だけが動き、何も公開しない。タグを選ぶと公開まで進む。

## タグの付け方

タグは、頭に `v` を付けるほかは [Semantic Versioning](https://semver.org/lang/ja/) に従う。どのタグで何が起きるかは、release.yml のトリガーとジョブの条件で決まる。

| タグの例 | パッケージの版 | GitHub Release | PyPI | TestPyPI |
| --- | --- | --- | --- | --- |
| `v1.0.0` | `1.0.0` | 通常のリリース | 公開する | 公開する |
| `v1.0.0-rc.1` | `1.0.0rc1` | pre-release | 公開する | 公開する |
| `v1.0.0-alpha.1` | `1.0.0a1` | pre-release | 公開しない | 公開する |
| `v1.0.0-beta.1` | `1.0.0b1` | pre-release | 公開しない | 公開する |

rc は必ず `-rc.N` の形で書く。`v1.0.0rc1` のように `-` を入れないと、トリガー（`v[0-9]+.[0-9]+.[0-9]+` と `v[0-9]+.[0-9]+.[0-9]+-*`）のどちらにも一致せず、ワークフローが起動しない。

パッケージの版は、setuptools-scm がタグを PEP 440 の形に正規化したものになる。`v1.0.0-rc.1` のタグからは `1.0.0rc1` ができ、PyPI 上の版も `__version__` もこの値になる。PyPI に出した rc は、pip が既定では選ばない（`pip install --pre` を付けたときか、版を明示したときだけ入る）。

## 最初のリリースは v1.0.0

現在の公開版は 0.1.0 で、次のリリースには API の破壊的変更が入る（内容は `CHANGELOG.md` の 1.0.0 の節を参照）。そのため、次の版は `v1.0.0` とする。

trusted publisher の設定が正しいかを先に確かめたい場合は、`v1.0.0-rc.1` を push してから `v1.0.0` を出してもよい。rc も PyPI に公開されるが、上のとおり pip は既定では rc を入れないので、利用者には影響しない。

## タグを push する前に版を確かめる

あるコミットにタグを付けたときにどの版ができるかは、一時的なクローンで確かめられる。タグはクローンにだけ付くので、作業中のリポジトリのタグは増えない。クローンにはコミット済みの内容だけが入るので、CI と同じ条件でビルドできる。リポジトリのルートで、`build` を入れた環境（`pip install ".[dev]"`）から実行する。

```sh
tmp="$(mktemp -d)"
git clone --quiet . "$tmp/repo"
git -C "$tmp/repo" tag v1.0.0
(cd "$tmp/repo" && python -m build --outdir "$tmp/dist")
ls "$tmp/dist"
unzip -p "$tmp"/dist/*.whl '*.dist-info/METADATA' | grep '^Version:'
rm -rf "$tmp"
```

`ensemble_outlier_sample_detection-1.0.0.tar.gz` と `ensemble_outlier_sample_detection-1.0.0-py3-none-any.whl` ができ、`Version: 1.0.0` と出れば正しい。`git clone .` は、いまチェックアウトしているブランチをクローンする。

タグを使わずに、指定した版でビルドが通るかだけを見るなら、環境変数 `SETUPTOOLS_SCM_PRETEND_VERSION` で版を与える。この方法はタグを読まないので、タグの書き方が正しいかは確かめられない。

```sh
SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 python -m build --outdir dist
```

## 失敗したとき

`build` が失敗した場合は、ほかのジョブは動かず、何も公開されていない。原因を直したコミットを main に入れてから、タグを消して付け直す。

```sh
git push --delete origin v1.0.0
git tag -d v1.0.0
```

PyPI と TestPyPI に一度上げた版は、削除しても同じ版で上げ直せない。公開したあとで問題が見つかった場合は、次の版（例えば `v1.0.1`）を出す。ネットワークなどの一時的なエラーで一部のジョブだけが落ちた場合は、Actions の Re-run failed jobs でそのジョブをやり直す。最初のリリースで起きやすいのは、trusted publisher の登録の誤り（owner、repository、workflow、environment の綴り）で公開ジョブだけが落ちるケースである。この場合も、PyPI / TestPyPI 側の登録を直してから Re-run failed jobs で公開ジョブをやり直せばよい。build が成功していればタグを付け直す必要はない。
