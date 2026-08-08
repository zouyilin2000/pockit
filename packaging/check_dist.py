"""Validate the contents of pockit's wheel and source distribution."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def _single_archive(dist_dir: Path, pattern: str) -> Path:
    archives = sorted(dist_dir.glob(pattern))
    if len(archives) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} archive in {dist_dir}, found {len(archives)}"
        )
    return archives[0]


def _expected_sdist_files(root: Path) -> set[PurePosixPath]:
    expected = {
        PurePosixPath("LICENSE"),
        PurePosixPath("README.md"),
        PurePosixPath("RELEASING.md"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("images/lqr_readme.png"),
    }
    patterns = {
        "pockit": ("*.py",),
        "tests": ("*.py",),
        "examples": ("*.py", "README.md"),
        "packaging": ("*.py", "*.md", "*.template"),
    }
    for directory, globs in patterns.items():
        for pattern in globs:
            for path in (root / directory).rglob(pattern):
                if path.is_file() and "__pycache__" not in path.parts:
                    expected.add(PurePosixPath(path.relative_to(root).as_posix()))
    return expected


def _check_wheel(wheel: Path, root: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = {PurePosixPath(name) for name in archive.namelist()}

    expected_package_files = {
        PurePosixPath(path.relative_to(root).as_posix())
        for path in (root / "pockit").rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    }
    missing_package_files = sorted(expected_package_files - names, key=str)
    if missing_package_files:
        raise RuntimeError(f"wheel is missing package files: {missing_package_files}")
    forbidden = ("examples/", "tests/", "packaging/", "images/")
    leaked = sorted(str(name) for name in names if str(name).startswith(forbidden))
    if leaked:
        raise RuntimeError(f"wheel contains source-only files: {leaked}")
    if not any(name.match("*.dist-info/licenses/LICENSE") for name in names):
        raise RuntimeError("wheel does not contain LICENSE in distribution metadata")


def _check_sdist(sdist: Path, root: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = [PurePosixPath(name) for name in archive.getnames()]

    roots = {name.parts[0] for name in names if name.parts}
    if len(roots) != 1:
        raise RuntimeError(f"sdist must have one top-level directory, found {roots}")
    relative_names = {
        PurePosixPath(*name.parts[1:]) for name in names if len(name.parts) > 1
    }
    missing = sorted(_expected_sdist_files(root) - relative_names, key=str)
    if missing:
        raise RuntimeError(f"sdist is missing maintained files: {missing}")
    generated = sorted(
        str(name)
        for name in relative_names
        if "__pycache__" in name.parts or name.suffix in {".pyc", ".pyo"}
    )
    if generated:
        raise RuntimeError(f"sdist contains generated Python files: {generated}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dist_dir = args.dist_dir.resolve()
    wheel = _single_archive(dist_dir, "*.whl")
    sdist = _single_archive(dist_dir, "*.tar.gz")
    _check_wheel(wheel, root)
    _check_sdist(sdist, root)
    print(f"Validated {wheel.name} and {sdist.name}")


if __name__ == "__main__":
    main()
