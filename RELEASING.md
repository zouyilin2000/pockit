# Release Guide / 发布指南

The distribution name is `pockit-optimal-control`; the Python import name remains
`pockit`. The shorter PyPI name `pockit` belongs to an unrelated project and must
not be used for this distribution.

发行包名称是 `pockit-optimal-control`，Python 导入名称仍为 `pockit`。PyPI 上较短的
`pockit` 名称属于另一个无关项目，因此本项目不能使用该发行名称。

## PyPI Trusted Publishing / PyPI 可信发布

Production releases use GitHub Actions and PyPI Trusted Publishing. Do not commit
a `.pypirc`, API token, or password, and do not add a production upload token to
GitHub Secrets.

正式版本通过 GitHub Actions 和 PyPI Trusted Publishing 发布。不要提交 `.pypirc`、
API token 或密码，也不要在 GitHub Secrets 中添加正式发布 token。

Before the first release:

1. In the GitHub repository settings, create an environment named `pypi`.
   Requiring a reviewer for this environment is recommended.
2. In PyPI account settings, add a pending trusted publisher with these exact
   values:
   - PyPI project name: `pockit-optimal-control`
   - GitHub owner: `zouyilin2000`
   - Repository: `pockit`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. Confirm that `.github/workflows/publish.yml` is present on the default branch.

首次发布前：

1. 在 GitHub 仓库设置中创建名为 `pypi` 的 environment，建议要求人工审批。
2. 在 PyPI 账户设置中添加 pending trusted publisher，并逐字填写上述项目名、所有者、
   仓库、workflow 和 environment。这个配置会预留新项目名并允许首次发布。
3. 确认 `publish.yml` 已经存在于默认分支。

## Build And Check / 构建与检查

Start from a clean checkout containing only files intended for the release. Update
`project.version` in `pyproject.toml`, then run the following commands in the
`pockit-dev` Conda environment. Ensure `dist/` contains no artifacts from an older
version before building.

从干净的检出开始，只保留需要发布的文件。更新 `pyproject.toml` 中的
`project.version`，确认 `dist/` 没有旧版本产物，然后执行：

```powershell
conda run -n pockit-dev python -m pytest -q
conda run -n pockit-dev python -m build
conda run -n pockit-dev python packaging/check_dist.py
conda run -n pockit-dev python -m twine check --strict dist/*
```

Inspect both archives. The wheel must contain only the `pockit` package and its
distribution metadata. The sdist must also contain `tests/`, `examples/`, `images/`,
and `packaging/`. Maintainer-only release instructions in `RELEASING.md` remain in
the Git repository and are intentionally excluded from distribution archives.

检查两个归档：wheel 只应包含 `pockit` 包和发行元数据；sdist 还必须包含测试、示例、
图片和 conda-forge 草案。仅供维护者使用的 `RELEASING.md` 保留在 Git 仓库中，且有意
不放入发行归档。

```powershell
$wheel = (Get-ChildItem dist -Filter *.whl).FullName
$sdist = (Get-ChildItem dist -Filter *.tar.gz).FullName
python -m zipfile -l $wheel
tar -tf $sdist
```

## Publish A Release / 发布版本

1. Commit the version change and wait for all CI jobs to pass.
2. Create an annotated tag whose value exactly matches the version, prefixed by
   `v`, for example `v0.1.0`.
3. Push the tag and publish a GitHub Release for that tag.
4. The `Publish to PyPI` workflow verifies the exact `vX.Y.Z` tag, builds fresh
   artifacts, validates them with Twine, and publishes through OIDC.
5. Verify the PyPI page and install the release in a new environment.

版本号一经上传到 PyPI 就不可覆盖。若 workflow 失败，应修复问题并发布新版本，不能
用同一版本替换已有文件。

```powershell
python -m venv .venv-release-check
.\.venv-release-check\Scripts\python -m pip install --upgrade pip
.\.venv-release-check\Scripts\python -m pip install pockit-optimal-control
.\.venv-release-check\Scripts\python -c "import pockit; from pockit.lobatto import System; from pockit.optimizer import scipy"
```

Ipopt and plotting support are optional:

```powershell
python -m pip install "pockit-optimal-control[ipopt,examples]"
```

`cyipopt` may require a system Ipopt installation. Conda-forge is generally the
simpler route for that native dependency.

## Conda-Forge / Conda-Forge 发布

The repository file `packaging/conda-forge/recipe.yaml.template` is only a draft for
the initial staged-recipes pull request. It is intentionally not publishable: the
SHA-256 value is a placeholder until the exact PyPI sdist exists. Do not treat this
repository as an official feedstock.

仓库中的 `packaging/conda-forge/recipe.yaml.template` 只是首次提交 staged-recipes
的草案，不是正式 feedstock。SHA-256 在 PyPI sdist 生成前无法确定，因此模板不能
直接构建或发布。

After the PyPI release:

1. Download the exact `pockit_optimal_control-X.Y.Z.tar.gz` file from PyPI.
2. Calculate its SHA-256 checksum and replace both the version and checksum in the
   template.
3. Fork `conda-forge/staged-recipes` and copy the completed recipe to the layout
   requested by its current contribution guide:
   `recipes/pockit-optimal-control/recipe.yaml`.
4. Confirm that runtime dependencies match the base dependencies in
   `pyproject.toml`. The Conda package also includes `cyipopt` so the Ipopt backend
   works by default; `matplotlib` remains optional.
5. Run the staged-recipes lint/build instructions, push the branch, and open a PR.
6. After merge, accept the maintainer invitation. The generated
   `conda-forge/pockit-optimal-control-feedstock` repository becomes the sole
   source of truth for the Conda recipe and its conda-smithy configuration.

发布 PyPI 后，下载对应 sdist，计算哈希并替换模板中的占位符；随后把完成的 recipe
提交到 `conda-forge/staged-recipes`。合并后，所有 feedstock 配置都只在新生成的
`conda-forge/pockit-optimal-control-feedstock` 仓库维护，本项目仓库不应复制
`.ci_support/`、`conda-forge.yml` 或 conda-smithy 生成的 workflow。

Calculate the checksum in the directory containing the downloaded sdist:

在下载 sdist 所在的目录中计算哈希：

```powershell
$sdist = "pockit_optimal_control-0.1.1.tar.gz"
(Get-FileHash $sdist -Algorithm SHA256).Hash.ToLowerInvariant()
```

Run the following commands from a clone of `conda-forge/staged-recipes`, after
creating a feature branch and copying the completed recipe into it. The official
repository's `build-locally.py` script is the authoritative local build path for
the current v1 recipe format.

填写并复制 recipe 后，在 `conda-forge/staged-recipes` 的功能分支根目录中执行以下命令。
对于当前 v1 recipe 格式，应以官方仓库中的 `build-locally.py` 为本地构建入口。

```powershell
conda create -n conda-forge-recipe -c conda-forge conda-smithy shellcheck
conda run -n conda-forge-recipe conda-smithy recipe-lint --conda-forge recipes/pockit-optimal-control
python build-locally.py
```

The Conda package and PyPI distribution are both named `pockit-optimal-control`.
The Python import name remains `pockit`. The Conda package installs `cyipopt` and
its native Ipopt dependency by default. Users who need the plotting dependency
can install it alongside Pockit:

```powershell
conda install -c conda-forge pockit-optimal-control matplotlib
```
