from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
BLOCKED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove MamaCare runtime caches and temporary files.")
    parser.add_argument(
        "--remove-venv",
        action="store_true",
        help="Also remove the local .venv directory. It can be recreated from requirements.txt.",
    )
    args = parser.parse_args()
    blocked_dirs = CACHE_DIRS | ({".venv"} if args.remove_venv else set())
    removed = []

    for path in sorted(ROOT.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name in blocked_dirs:
            shutil.rmtree(path, ignore_errors=False)
            removed.append(path.relative_to(ROOT).as_posix() + "/")
        elif path.is_file() and path.suffix.lower() in BLOCKED_SUFFIXES:
            path.unlink()
            removed.append(path.relative_to(ROOT).as_posix())

    print(f"Removed {len(removed)} runtime cache or temporary paths.")
    if not args.remove_venv:
        print("The ignored .venv environment was preserved; final packaging excludes it automatically.")


if __name__ == "__main__":
    main()
