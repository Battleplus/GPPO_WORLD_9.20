"""Run the weak-communication contract tests without installing pytest."""

from __future__ import annotations

import runpy
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    paths = (root / "tests" / "test_m10_environment.py", root / "tests" / "test_m10_weak_communication.py")
    executed = []
    for path in paths:
        namespace = runpy.run_path(str(path))
        functions = [(name, value) for name, value in namespace.items()
                     if name.startswith("test_") and callable(value)]
        for name, function in functions:
            function()
            executed.append(f"{path.name}::{name}")
    print(f"weak_communication_contract_passed={len(executed)}")
    for name in executed:
        print(name)


if __name__ == "__main__":
    main()
