"""Copy the web UI source from ../ui into the package static folder.

Run after editing the UI and before building/testing the package.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ui"
TARGET = Path(__file__).resolve().parent / "desireeiaserver" / "static"

_IGNORED = {".DS_Store", "*.pyc"}


def _ignore_dir(directory: Path, names: list[str]) -> set[str]:
    return {n for n in names if n in _IGNORED}


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"UI source not found: {SOURCE}")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    shutil.copytree(SOURCE, TARGET, ignore=_ignore_dir, dirs_exist_ok=True)
    print(f"synced {SOURCE} -> {TARGET}")


if __name__ == "__main__":
    main()