"""Launch a pre-specified CS01/E02/v001 split with visible runner progress.

Run one seed per terminal for parallel execution, or use ``all`` to run the
three seeds sequentially in one terminal. Every attempt gets a fresh run ID.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "projects/context_selection/src/multi_dataset_context.py"
SAMPLE_SEEDS = {101: 301, 102: 302, 103: 303}


def command(seed: int) -> list[str]:
    return [sys.executable, "-u", str(RUNNER), "--run",
            "--datasets", "Hillstrom", "Retailhero-uplift", "LZD",
            "Criteo-ITE-v2.1", "--out-dir",
            str(ROOT / "artifacts/local/cs01_e02_results" / f"split_{seed}"),
            "--device", "cpu", "--folds", "1",
            "--split-seed", str(seed),
            "--criteo-sample-seed", str(SAMPLE_SEEDS[seed]),
            "--context-sizes", "512", "--context-seeds", "201", "202", "203",
            "--train-max", "51200", "--eval-max", "4096",
            "--criteo-sample-max", "100000", "--bootstrap", "500",
            "--effect-score-mode", "crossfit", "--nuisance-folds", "5",
            "--nuisance-seed", "42", "--threads", "4",
            "--revision", "83aad07da1cb077cfda4236878a1b07dc9f72a54"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", choices=("101", "102", "103", "all"),
                        help="Outer split seed; 'all' runs 101, 102, 103 sequentially")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the exact command without starting CausalPFN")
    args = parser.parse_args()
    seeds = tuple(SAMPLE_SEEDS) if args.seed == "all" else (int(args.seed),)
    env = os.environ.copy()
    env.pop("TQDM_DISABLE", None)
    cache = ROOT / "artifacts/local/cache"
    if not args.dry_run:
        (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache / "matplotlib")
    env["XDG_CACHE_HOME"] = str(cache)
    for seed in seeds:
        cmd = command(seed)
        print(f"CS01/E02: split seed {seed}, Criteo sample seed {SAMPLE_SEEDS[seed]}",
              flush=True)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if result.returncode:
            print(f"Split {seed} stopped with exit code {result.returncode}; "
                  "inspect its local output before retrying.", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
