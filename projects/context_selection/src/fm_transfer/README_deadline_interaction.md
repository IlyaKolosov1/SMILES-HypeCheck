# Bounded supplementary Criteo interaction

This adds one comparison to the existing paper without rerunning Do-PFN.
It uses a previously opened holdout, so it is a **postselection supplementary
analysis**, not an independent confirmation of a newly registered hypothesis.

The fixed design is Criteo `visit`, K=1024, T versus T+Y, seeds 42/43/44,
and 50,000 identical evaluation rows. Only six CausalPFN configurations run.
The exact contexts and their row order come from the corresponding saved
Do-PFN prediction files. The eval subset is selected by T/Y stratification
with seed 42 before inspecting scores. No choice depends on a favorable result.

The Do-PFN NPZ/JSON snapshot supplied in TRANSFER differs slightly from the
older manuscript CSV. The runner verifies each source NPZ against its own JSON
and recomputes all subset metrics. Do not combine this comparison with the old
table estimates or bootstrap intervals.

## Inputs

- Original `Criteo-ITE-v2.1/features.parquet` and `outcomes.parquet`. Their exact
  SHA256 values are pinned in the runner. `--data-root` accepts either this
  directory or its parent `data_A_cleaned`.
- Six pairs of `holdout_dopfn_{strategy}_ctx1024_seed{seed}_evalall_cuda0.json`
  and `.predictions.npz`. Strategies are `balanced_treatment` and
  `balanced_treatment_outcome`, seeds 42/43/44. They total approximately 67 MiB.
- For inference, a local `causalpfn_v0.pt` checkpoint and an environment with
  **causalpfn==0.1.5**, numpy, pandas, pyarrow, scikit-learn,
  scikit-uplift==0.5.1, tqdm and PyTorch. CUDA is the default; CPU is explicitly
  supported with `--device cpu`. MPS is not supported by this runner.

Reuse an existing compatible package environment. The runner does not
install packages or silently fall back to another device. It records package versions,
CausalPFN Python-source hash, input hashes, checkpoint hash, batch size and
per-arm device details and CPU thread count. The historical Do-PFN checkpoint
itself is not reloaded. Using CPU here and saved GPU Do-PFN predictions is
reported as a device difference, not a measurement of runtime superiority.

## Commands

Run from the repository root, replacing paths with the remote machine paths.

```bash
python -u projects/context_selection/src/fm_transfer/run_deadline_interaction.py \
  --check \
  --data-root /path/to/data_A_cleaned \
  --dopfn-output /path/to/fm_holdout/criteo \
  --output /path/to/results/criteo_interaction_50k
```

`--check` requires no GPU and does not import the model or write scores. It
checks source hashes, complete holdout identity, label alignment, six contexts,
and the source Qini/uplift@10% values. It streams the Parquet files and retains
only needed feature rows in RAM. No old server cache or split cache is needed.

```bash
python -u projects/context_selection/src/fm_transfer/run_deadline_interaction.py \
  --run \
  --data-root /path/to/data_A_cleaned \
  --dopfn-output /path/to/fm_holdout/criteo \
  --checkpoint /path/to/causalpfn_v0.pt \
  --output /path/to/results/criteo_interaction_50k \
  --max-hours 2
```

The worker uses the original context-fitted preprocessing and CausalPFN
defaults with batch size 128. It freezes `protocol.json` before new inference.
It saves only completed arms, estimates remaining cost after the first pair,
and stops if the estimate exceeds the remaining wall-time allowance. Each
worker subprocess has a timeout using the remaining allowance. Budget exit
code is 2. Rerun the same command to resume verified completed arms. An arm
interrupted before its completion JSON is recomputed in full. `--max-hours`
allows 1, 2, 3, 4, or 6 hours for an invocation without changing the scientific
plan. This is a time limit, not a promised runtime. Device, thread count,
checkpoint and source changes require a fresh output directory.

## Local CPU benchmark and launch

Before a local run, use `--benchmark` with the same data, checkpoint, output,
device and thread arguments as `--run`. This freezes the same protocol and
checks the first treatment-balanced and outcome-balanced contexts at seed 42.
The first 128 queries must agree within rtol 2e-4 / atol 2e-5 under
repetition and under two batches of 64. Four batches of 128 from a fixed prefix
are then timed for each context. The benchmark computes no uplift metrics and
does not select a cohort or method based on observed performance.

Example for the current Mac checkout (run from repository root):

```bash
.venv/bin/python -u projects/context_selection/src/fm_transfer/run_deadline_interaction.py \
  --benchmark --device cpu --cpu-threads 2 \
  --data-root artifacts/local/data_A_cleaned \
  --dopfn-output '/Users/ilya-kolosov/Downloads/TRANSFER/archives/code 2/studies/home/projects/smiles/results/fm_holdout/criteo' \
  --checkpoint artifacts/local/cache/huggingface/hub/models--vdblm--causalpfn/snapshots/83aad07da1cb077cfda4236878a1b07dc9f72a54/causalpfn_v0.pt \
  --output artifacts/local/deadline_interaction_20261116_cpu --max-hours 2
```

After checking `benchmark.json`, replace `--benchmark` with `--run` to run all
six arms and the paired bootstrap. On macOS, prefix the command with
`caffeinate -i` to prevent idle system sleep while it runs; keep the lid open.
The benchmark has a maximum 30-minute subprocess allowance. This is ample for
a speed probe and prevents a mistaken configuration from consuming the night.

Measured on 2026-11-16: Apple M4 Pro, 24 GiB, PyTorch 2.3.1, 2 CPU threads;
128-query batches took 0.331–0.350 s. Repeated scores were identical and changing
128 to two 64-query batches changed scores by at most 2.09e-7. The six-arm timing
projection including a 600-second analysis reserve was 23.8 minutes. Actual time
can differ; the full inference retains its explicit wall-time cap.

The local weight file was resolved offline from
[`vdblm/causalpfn` at revision `83aad07da1cb077cfda4236878a1b07dc9f72a54`](https://huggingface.co/vdblm/causalpfn/tree/83aad07da1cb077cfda4236878a1b07dc9f72a54).
Its `causalpfn_v0.pt` symlink points to cache blob
`4f0f53717f1c62426fe98faa0459d33bc8f9862a87e8484ca3869ca899e7841d`,
which is also its verified SHA256. No model download was required. The local
CausalPFN Python-source SHA256 is
`8de2fdb3d72334e00aa57e6de1e19095a9d769a5b8c6f3e010f3b3d918e51e8b`.
All 11 installed Python files were verified byte-for-byte against the official
PyPI `causalpfn==0.1.5` wheel, whose SHA256 is
`c129280a40bdf1b5639ec9ecdf91a308f539c80d93990a9d997b886cd9a08d8f`.
In particular, the capped-k `nearest_indices_1d` helper is part of that
unmodified distribution; it is not a local inference patch. The wheel is
preserved in `artifacts/local/cache/wheels/` and the per-file comparison,
official download URL and PyPI release timestamp are recorded separately in
`package_distribution_audit.json` beside the run outputs.
These facts identify the new supplementary run; they do not by themselves
establish byte identity with the historical server's CausalPFN installation.

After all six arms finish, analysis runs automatically. If the time budget is
exhausted during bootstrap, use the CPU-only command below. Partial confidence
intervals are never reported.

```bash
python -u projects/context_selection/src/fm_transfer/run_deadline_interaction.py \
  --analyze --output /path/to/results/criteo_interaction_50k --max-hours 2
```

## Interpretation

Primary metric is normalized Qini AUC. Secondary metric is uplift@10% in
percentage points. For each seed compute the paired T+Y minus T effect for
each model, then average seeds. The primary interaction is

```
(Do-PFN_TY − Do-PFN_T) − (CausalPFN_TY − CausalPFN_T)
```

A positive interaction means T+Y has a more favorable relative effect on
Do-PFN in this fixed comparison. It does not by itself prove a negative effect
on CausalPFN or superiority of Do-PFN in absolute score. Report both component
differences along with the interaction.

The 1,000 paired bootstrap draws resample identical raw feature-vector groups.
Each draw uses the same evaluation multiplicities for all 12 score vectors.
These feature hashes are proxy clusters, not verified customer identifiers.
The intervals are conditional on fitted models, contexts, preprocessing,
the opened holdout and the fixed subset. They do not include training, split,
context-selection or seed-sampling uncertainty. Three seeds are not three
independent datasets. Intervals are exploratory and pointwise.

Outputs include `interaction_summary.csv`, `analysis.json`, the frozen
protocol, prepared input, six CausalPFN score files with provenance JSON, and
`bootstrap_draws.npz`. Do not report results until `analysis.json` has status
`complete`. No output is uploaded automatically.

## Local validation

```bash
python -B -m unittest discover \
  -s projects/context_selection/src/fm_transfer/tests \
  -p 'test_deadline_interaction.py' -v
```

Tests cover pairing, context leakage, changed provenance, the interaction sign,
and exact cancellation for identical models under paired cluster resampling.
The CPU smoke test also rejects materially batch-dependent or nonrepeatable
scores. Unit tests do not substitute for `--benchmark` on the selected device.


## Completed local comparison (2026-11-16)

The frozen 50,000-row run completed all six CausalPFN arms and all 1,000
paired bootstrap draws. There were 49,825 distinct raw-feature groups.
New worker time summed to 827.501 seconds; the full invocation took about
974 seconds (16.2 minutes) including preparation and bootstrap, measured from
its fresh log creation to final output. This excludes installation and the
preceding benchmark, and does not include the historical Do-PFN inference.

| Contrast | Qini change | Conditional 95% interval |
| --- | ---: | --- |
| Do-PFN T+Y minus T | +0.096063 | [0.033748, 0.150849] |
| CausalPFN T+Y minus T | +0.034380 | [0.005156, 0.065666] |
| Difference: Do-PFN minus CausalPFN | +0.061683 | [-0.003406, 0.123251] |

Both Qini component changes are positive, but their difference remains
unresolved. This does not establish opposite model responses or failure of
balancing to transfer to CausalPFN. The top-decile changes are +1.2429 pp for
Do-PFN (interval [-1.1793, 3.4226]) and -1.0372 pp for CausalPFN
([-2.9983, 0.9530]); both intervals include zero. These are conditional,
postselection estimates on an already opened holdout, not independent
confirmation. Do not combine them with the original full-holdout intervals.

Generate the verified vector figure after a complete run:

```bash
python -B projects/context_selection/src/fm_transfer/plot_deadline_interaction.py \
  --run-dir artifacts/local/deadline_interaction_20261116_cpu \
  --output-dir artifacts/local/deadline_interaction_20261116_cpu/figures
```

The plotter verifies source hashes, all twelve score vectors, metric values,
and bootstrap quantiles before generating PDF, PNG, CSV, and metadata JSON.
