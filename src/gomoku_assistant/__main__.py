from __future__ import annotations

import sys


def main() -> int:
    try:
        from .ui import run
    except ModuleNotFoundError as error:
        if error.name == "PySide6":
            print("PySide6 is required. Install project dependencies before starting the desktop app.")
            return 2
        raise
    return run()


if __name__ == "__main__":
    sys.exit(main())

