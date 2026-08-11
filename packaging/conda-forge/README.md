# Conda-Forge Recipe Draft

This directory is an upstream staging aid, not an official conda-forge feedstock.
The authoritative recipe will live in
`conda-forge/pockit-optimal-control-feedstock` only after the initial
`conda-forge/staged-recipes` pull request is accepted.

`recipe.yaml.template` uses conda-forge's required v1 recipe format and deliberately
contains `REPLACE_WITH_PYPI_SDIST_SHA256`. It cannot be built or published until
`pockit-optimal-control` has been released on PyPI and the checksum of that exact
source archive has replaced the placeholder.

Do not add generated feedstock files such as `.ci_support/`, `conda-forge.yml`, or
conda-smithy workflows to this repository. See [RELEASING.md](../../RELEASING.md)
for the complete release and staged-recipes procedure.

此目录只是上游项目准备首次 recipe 的辅助材料，并不是正式 conda-forge feedstock。
`recipe.yaml.template` 采用 conda-forge 要求的 v1 recipe 格式。模板中的 SHA-256 是
有意保留的占位符；必须先发布 PyPI sdist，再填写对应版本与哈希，然后按 conda-forge
当前贡献指南提交到 `staged-recipes`。合并后，正式 recipe 只在独立的
`conda-forge/pockit-optimal-control-feedstock` 仓库维护。
