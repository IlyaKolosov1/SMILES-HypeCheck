"""Launch the fresh-row Criteo CS01/E03/v001 experiment with concise progress."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "projects/context_selection/src/cs01_e03.py"


if __name__ == "__main__":
    raise SystemExit(subprocess.run([sys.executable, "-u", str(RUNNER),
                                    *sys.argv[1:]], cwd=ROOT, check=False).returncode)
