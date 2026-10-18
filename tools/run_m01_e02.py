"""Launch M01/E02/v001 with progress, or verify inputs without training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "projects/model_comparison/src/m01_e02.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", choices=("101", "102", "103", "all"),
                        default="all", help="One CS01 split or all three in order")
    parser.add_argument("--check", action="store_true",
                        help="Verify source hashes, rows and labels without training or writing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the runner command without executing it")
    args = parser.parse_args()
    command = [sys.executable, "-u", str(RUNNER), args.seed]
    if args.check:
        command.append("--check")
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
