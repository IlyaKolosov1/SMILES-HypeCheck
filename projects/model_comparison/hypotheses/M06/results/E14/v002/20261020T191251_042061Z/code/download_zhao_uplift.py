"""Download and verify Zhao's public uplift benchmark from Zenodo.

The CSV stays in artifacts/local and is never committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/local/external/zhao_uplift/synthetic_uplift_data.csv"
URL = "https://zenodo.org/records/6342553/files/synthetic_uplift_data.csv?download=1"
SIZE = 785_863_095
MD5 = "48101d5fa9fac9ea530b81375108afb6"


def digest(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - Zenodo publishes MD5 as a file checksum.
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def download_part(index: int, start: int, end: int, folder: Path) -> Path:
    part = folder / f"part_{index:03d}"
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return part
    temporary = part.with_suffix(".partial")
    attempt_path = folder / f"part_{index:03d}.attempt"
    existing = temporary.stat().st_size if temporary.exists() else 0
    if existing > expected:
        temporary.unlink()
        existing = 0
    for attempt in range(12):
        current = start + existing
        attempt_path.unlink(missing_ok=True)
        command = [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--max-time", "30", "--range", f"{current}-{end}",
            "--output", str(attempt_path), URL,
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        received = attempt_path.stat().st_size if attempt_path.exists() else 0
        if received:
            with temporary.open("ab") as output, attempt_path.open("rb") as data:
                shutil.copyfileobj(data, output)
            existing += received
            attempt_path.unlink(missing_ok=True)
        if existing == expected:
            break
        if existing > expected:
            raise RuntimeError(f"Part {index} exceeded expected length")
        if result.returncode not in (0, 28, 18):
            raise RuntimeError(f"Part {index} failed: {result.stderr.strip()}")
        time.sleep(min(2 + attempt, 10))
    else:
        raise TimeoutError(f"Part {index} incomplete after 12 range-resume attempts")
    if temporary.stat().st_size != expected:
        raise RuntimeError(f"Part {index} has {temporary.stat().st_size} bytes; expected {expected}")
    temporary.replace(part)
    return part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing CSV without network access")
    parser.add_argument("--workers", type=int, default=24, help="Parallel downloads (default: 24)")
    parser.add_argument("--parts", type=int, default=256, help="Range count (default: 256)")
    args = parser.parse_args()
    if args.workers < 1 or args.parts < 1:
        parser.error("--workers and --parts must be positive")

    if DEST.exists() and DEST.stat().st_size == SIZE:
        md5, sha256 = digest(DEST)
        if md5 == MD5:
            print(f"VERIFIED {DEST}\nsize={SIZE} md5={md5} sha256={sha256}", flush=True)
            return
        if args.check:
            raise RuntimeError(f"Checksum mismatch: {md5}, expected {MD5}")
    if args.check:
        raise FileNotFoundError(f"Verified file is absent: {DEST}")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    folder = DEST.parent / ".download_parts"
    folder.mkdir(exist_ok=True)
    ranges = [
        (i, SIZE * i // args.parts, SIZE * (i + 1) // args.parts - 1)
        for i in range(args.parts)
    ]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_ranges = {pool.submit(download_part, i, start, end, folder): (start, end)
                         for i, start, end in ranges}
        pending = set(future_ranges)
        last_print = 0.0
        while pending:
            finished = {future for future in pending if future.done()}
            for future in finished:
                future.result()
            pending -= finished
            now = time.monotonic()
            if now - last_print >= (1.0 if sys.stderr.isatty() else 20.0) or not pending:
                completed = [future for future in future_ranges
                             if future.done() and future.exception() is None]
                received = sum(future_ranges[future][1]-future_ranges[future][0]+1
                               for future in completed)
                elapsed = max(now - started, 0.001)
                speed = received / elapsed
                eta = (SIZE - received) / speed if speed else float("inf")
                line = (f"Zenodo download {received / SIZE:5.1%} "
                        f"({received / 1e6:.0f}/{SIZE / 1e6:.0f} MB), "
                        f"{len(completed)}/{len(future_ranges)} parts, "
                        f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m")
                print(line, end="\r" if sys.stderr.isatty() else "\n", file=sys.stderr, flush=True)
                last_print = now
            if pending:
                time.sleep(0.5)
    if sys.stderr.isatty():
        print(file=sys.stderr)

    assembled = DEST.with_suffix(".csv.partial")
    with assembled.open("wb") as output:
        for index, _, _ in ranges:
            with (folder / f"part_{index:03d}").open("rb") as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    if assembled.stat().st_size != SIZE:
        raise RuntimeError("Assembled file has the wrong size")
    md5, sha256 = digest(assembled)
    if md5 != MD5:
        raise RuntimeError(f"Zenodo MD5 mismatch: {md5}, expected {MD5}")
    os.replace(assembled, DEST)
    shutil.rmtree(folder)
    print(f"VERIFIED {DEST}\nsize={SIZE} md5={md5} sha256={sha256}", flush=True)


if __name__ == "__main__":
    main()
