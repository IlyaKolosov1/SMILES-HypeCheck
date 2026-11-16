#!/usr/bin/env python3
"""Plot only a completed, verified supplementary model-by-context comparison.

No model inference or outcome-dependent selection is performed here. The input
is the fixed 50k Criteo run produced by run_deadline_interaction.py. This is an
analysis of an already opened holdout, not an independent confirmatory test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CONTRASTS = (
    "DoPFN_TY_minus_T",
    "CausalPFN_TY_minus_T",
    "interaction_Do_minus_Causal",
)
METRICS = ("normalized_qini_auc", "uplift10_pp")
STRATEGIES = ("balanced_treatment", "balanced_treatment_outcome")
STEM = "criteo_matched_context_interaction"
CAPTION = (
    "Supplementary comparison on a fixed 50,000-row subset of the already "
    "opened Criteo holdout at K=1,024. The first two rows show the change "
    "from treatment balancing (T) to treatment-and-outcome balancing (T+Y); "
    "the third subtracts the CausalPFN change from the Do-PFN change. "
    "Points average three context seeds. Bars are pointwise 95% paired "
    "feature-group bootstrap intervals conditional on the saved scores, "
    "contexts, and subset. Both pipelines receive the same ordered context "
    "rows in each pair. Reused Do-PFN scores are a separately versioned "
    "snapshot from the original full-holdout table."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load_verified(run_dir):
    """Verify provenance, point metrics and interval arithmetic before plotting."""
    from sklift.metrics import qini_auc_score, uplift_at_k

    run_dir = Path(run_dir).expanduser().resolve()
    cfg = json.loads((run_dir / "protocol.json").read_text())
    report = json.loads((run_dir / "analysis.json").read_text())
    ph = digest(run_dir / "protocol.json")
    require(report.get("status") == "complete", "Analysis is incomplete")
    require(report.get("protocol_sha256") == ph, "Analysis/protocol identity differs")
    require(cfg.get("analysis") == report.get("analysis") ==
            "supplementary_postselection_opened_holdout_interaction", "Unexpected analysis protocol")
    require(cfg.get("dataset") == "criteo" and cfg.get("outcome") == "visit", "Wrong dataset/outcome")
    require(cfg.get("context_size") == 1024, "Unexpected context budget")
    require(cfg.get("seeds") == [42, 43, 44], "Unexpected seeds")
    require(cfg.get("strategies") == list(STRATEGIES), "Strategy order differs")
    require(cfg.get("eval_size") == report.get("n_eval") == 50_000, "Unexpected evaluation size")
    require(cfg["bootstrap"]["draws"] == report.get("draws") == 1000,
            "The paper plot requires all 1000 planned bootstrap draws")
    require(cfg.get("source_reuse_not_original_paper_table") is True and
            report.get("source_reuse_not_original_paper_table") is True,
            "Do-PFN source snapshot distinction is missing")
    prep_hash = json.loads((run_dir / "prepared.sha256.json").read_text())["sha256"]
    require(digest(run_dir / "prepared.npz") == prep_hash, "Prepared inputs were modified")

    hashes = {name: digest(run_dir / name) for name in
              ("protocol.json", "analysis.json", "prepared.npz", "bootstrap_draws.npz")}
    # Recompute the points independently from the saved paired score vectors.
    actual_metrics = np.empty((2, 3, 2, 2))
    with np.load(run_dir / "prepared.npz", allow_pickle=False) as prep:
        y, t = prep["y"], prep["t"]
        require(y.shape == t.shape == (50_000,), "Invalid labels")
        require(len(np.unique(prep["groups"])) == report.get("n_feature_groups"),
                "Feature group count differs")
        for s, seed in enumerate((42, 43, 44)):
            for a, strategy in enumerate(STRATEGIES):
                label = f"{seed}_{strategy}"
                path = run_dir / f"causal_{label}.npz"
                record = json.loads(path.with_suffix(".json").read_text())
                require(record.get("label") == label and record.get("protocol_sha256") == ph,
                        f"Unpaired CausalPFN record: {label}")
                require(record.get("prediction_sha256") == digest(path),
                        f"Modified CausalPFN predictions: {label}")
                hashes[path.name] = digest(path)
                with np.load(path, allow_pickle=False) as c:
                    scores = [prep["dopfn_" + label], c["scores"]]
                for m, score in enumerate(scores):
                    require(score.shape == y.shape and np.isfinite(score).all(), "Invalid score vector")
                    actual_metrics[m, s, a] = (
                        qini_auc_score(y, score, t),
                        100 * uplift_at_k(y, score, t, strategy="overall", k=.1),
                    )

    with np.load(run_dir / "bootstrap_draws.npz", allow_pickle=False) as z:
        draws, point_metrics = z["draws"], z["point_metrics"]
    require(draws.shape == (1000, 3, 2) and np.isfinite(draws).all(), "Invalid bootstrap array")
    require(point_metrics.shape == (2, 3, 2, 2), "Unexpected metric array layout")
    np.testing.assert_allclose(point_metrics, actual_metrics, rtol=0, atol=1e-10,
                               err_msg="Stored point metrics disagree with saved scores")
    effects = (actual_metrics[:, :, 1] - actual_metrics[:, :, 0]).mean(axis=1)
    point = np.vstack((effects, effects[0] - effects[1]))
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    rows = {(r["contrast"], r["metric"]): r for r in report["results"]}
    require(len(rows) == len(report["results"]) == 6, "Missing or duplicate analysis contrasts")
    for c, contrast in enumerate(CONTRASTS):
        for m, metric in enumerate(METRICS):
            record = rows[contrast, metric]
            np.testing.assert_allclose(
                [record["point"], record["low95"], record["high95"]],
                [point[c, m], lo[c, m], hi[c, m]], rtol=0, atol=1e-12,
                err_msg=f"Analysis/bootstrap discrepancy: {contrast}, {metric}",
            )
    return [rows[c, METRICS[0]] for c in CONTRASTS], cfg, report, hashes


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    ink, grid, muted = "#223443", "#E3E9EE", "#5B6B78"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.7,
        "text.color": ink, "axes.labelcolor": ink, "xtick.color": ink,
        "ytick.color": ink, "axes.edgecolor": "#83919B",
        "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    fig.subplots_adjust(left=.325, right=.982, top=.84, bottom=.23)
    names = ("Do-PFN", "CausalPFN", "Difference\nDo - Causal")
    colors = ("#145F86", "#AA4D22", ink)
    markers = ("o", "s", "D")
    low = min(0., *(min(r["low95"], r["point"]) for r in rows))
    high = max(0., *(max(r["high95"], r["point"]) for r in rows))
    span = max(high - low, .02)
    ax.set_xlim(low - .09 * span, high + .09 * span)
    ax.set_ylim(-.35, 2.65)
    ax.axvline(0, color="#83919B", lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(.5, color=grid, lw=.7, zorder=0)
    for i, (r, color, marker) in enumerate(zip(rows, colors, markers)):
        y = 2 - i
        ax.plot([r["low95"], r["high95"]], [y, y], color=color, lw=1.5, zorder=3)
        ax.plot([r["low95"], r["high95"]], [y, y], "|", color=color, ms=4, mew=.85)
        ax.plot(r["point"], y, marker=marker, color=color, ms=4.4,
                markeredgecolor="white", markeredgewidth=.35, zorder=4)
        label = f'{r["point"]:+.3f} [{r["low95"]:+.3f}, {r["high95"]:+.3f}]'
        ax.text(.99, y + .25, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color=color)
    ax.set_yticks([2, 1, 0], names)
    ax.tick_params(axis="y", length=0, pad=5, labelsize=7.5)
    ax.tick_params(axis="x", length=2.3, width=.6, pad=3, labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Change in normalized Qini AUC", fontsize=7.7, labelpad=5)
    fig.text(.325, .955, "Criteo | matched contexts | K = 1,024", ha="left",
             va="top", fontsize=7.7, weight="medium")
    fig.text(.325, .89, "Seed means; conditional 95% intervals",
             ha="left", va="top", fontsize=7, color=muted)
    # Fixed dimensions keep labels consistent with the other manuscript figures.
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{STEM}.{extension}", dpi=300, facecolor="white",
                    metadata={"Creator": "Matplotlib"} if extension == "pdf" else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check saved results without creating figures")
    args = parser.parse_args()
    rows, cfg, report, hashes = load_verified(args.run_dir)
    if args.verify_only:
        print("PASS: source hashes, twelve score vectors, point metrics, and 1000 paired draws agree.")
        return
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    plot(rows, out)
    with (out / f"{STEM}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("contrast", "metric", "point", "low95", "high95"))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "analysis": cfg["analysis"], "n_eval": report["n_eval"],
        "context_size": cfg["context_size"], "seeds": cfg["seeds"],
        "n_feature_groups": report["n_feature_groups"], "bootstrap": cfg["bootstrap"],
        "source_reuse_not_original_paper_table": True,
        "input_sha256": hashes, "generator_sha256": digest(Path(__file__)),
        "dimensions_inches": [3.35, 2.15], "caption": CAPTION,
        "primary_rows": rows, "all_component_results": report["results"],
        "interpretation": "A positive interaction alone does not establish opposite responses or a three-model ranking reversal.",
    }
    (out / f"{STEM}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(CAPTION)
    print(f"Verified figure and source data: {out / STEM}")


if __name__ == "__main__":
    main()
