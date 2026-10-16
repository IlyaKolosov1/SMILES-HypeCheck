"""Check links and immutable result snapshots without loading datasets or FM weights."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
LINK = re.compile(r"\]\(([^)]+)\)")
STATUSES = {
    "plan", "smoke_only", "incomplete", "complete_exploratory",
    "historical_pilot", "historical_report_only", "reviewed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_run(run: Path, errors: list[str]) -> None:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"Missing manifest: {run.relative_to(ROOT)}")
        return
    try:
        record = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid manifest {manifest_path.relative_to(ROOT)}: {exc}")
        return
    hypothesis = run.parents[3]
    project = hypothesis.parents[1]
    expected = (project.name, hypothesis.name, run.parents[1].name,
                run.parent.name, run.name)
    actual = tuple(record.get(key) for key in (
        "project", "hypothesis", "experiment", "experiment_version", "run_id"))
    if actual != expected:
        errors.append(f"Path/manifest mismatch: {run.relative_to(ROOT)} {actual} != {expected}")
    if record.get("status") not in STATUSES:
        errors.append(f"Unknown status: {run.relative_to(ROOT)}")
    protocol = hypothesis / "experiments" / expected[2] / f"{expected[3]}.md"
    if not protocol.is_file():
        errors.append(f"Missing experiment version: {protocol.relative_to(ROOT)}")
    snapshot_name = record.get("code_snapshot")
    code_sha = record.get("code_sha256")
    if snapshot_name:
        snapshot = ROOT / snapshot_name
        if not snapshot.is_file() or sha256(snapshot) != code_sha:
            errors.append(f"Code snapshot mismatch: {run.relative_to(ROOT)}")
    listed = set()
    for item in record.get("files", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"Unsafe result path: {run.relative_to(ROOT)}/{relative}")
            continue
        file = run / relative
        listed.add(relative.as_posix())
        if not file.is_file() or sha256(file) != item.get("sha256") or file.stat().st_size != item.get("size"):
            errors.append(f"Result file mismatch: {run.relative_to(ROOT)}/{relative}")
    actual_files = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    unexpected = actual_files - listed - {"manifest.json", "README.md"}
    if unexpected:
        errors.append(f"Unlisted result files: {run.relative_to(ROOT)} {sorted(unexpected)}")


def main() -> int:
    errors: list[str] = []
    for markdown in [ROOT / "README.md", *ROOT.glob("docs/**/*.md"),
                     *ROOT.glob("projects/**/*.md"), *ROOT.glob("artifacts/*.md")]:
        for target in LINK.findall(markdown.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            local = markdown.parent / target.split("#", 1)[0]
            if not local.exists():
                errors.append(f"Broken link: {markdown.relative_to(ROOT)} -> {target}")
    runs = 0
    for project in PROJECTS.iterdir():
        hypotheses = project / "hypotheses"
        if not hypotheses.is_dir():
            continue
        for hypothesis in hypotheses.iterdir():
            if not hypothesis.is_dir():
                continue
            for version in (hypothesis / "results").glob("*/*"):
                for run in version.iterdir():
                    if run.is_dir():
                        runs += 1
                        check_run(run, errors)
    if errors:
        print("Repository validation failed:")
        for error in errors:
            print("-", error)
        return 1
    print(f"Repository validation passed: {runs} result snapshots and local links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
