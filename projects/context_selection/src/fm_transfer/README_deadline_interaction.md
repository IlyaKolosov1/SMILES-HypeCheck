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
- For inference, a local `causalpfn_v0.pt` checkpoint and a working CUDA
  environment with **causalpfn==0.1.5**, numpy, pandas, pyarrow, scikit-learn,
  scikit-uplift==0.5.1, tqdm and CUDA PyTorch.

The original GPU package environment should be reused. The runner does not
install packages or silently fall back to CPU. It records package versions,
CausalPFN Python-source hash, input hashes, checkpoint hash, batch size and
per-arm GPU details. The historical Do-PFN checkpoint itself is not reloaded.

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
interrupted before its completion JSON is recomputed in full. `--max-hours 3`
raises the allowance for that invocation without changing the scientific plan.

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
They do not substitute for a real CUDA inference check.
