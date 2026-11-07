# Matched T versus T+Y holdout for model transfer

This folder contains a snapshot of the supplied three-FM pipeline. The new
`run_transfer_holdout.py` schedules **12 missing Criteo jobs**: CausalPFN and
TabPFN-T, each with treatment-only and treatment+outcome context, seeds
42/43/44 and `K=1024`. The Do-PFN runs at the same K already exist in the
original study and are not silently recomputed or merged here.

The experiment reuses the original grouped 15% holdout. Because this holdout
has already been examined, these new runs are a **supplementary comparison**,
not independent confirmation. Do not label the resulting intervals as a
fresh, pre-registered test. The wrapper never overwrites successful jobs;
rerun the same command to resume after failure.

## Requirements

Use the original GPU environment that runs CausalPFN and TabPFN-T, the original
Criteo data and dense/group caches, and the original `fm_cv/criteo/fold0_*.json`
files. The vendored holdout script checks that the holdout hash agrees with
those CV files before model inference. Its output is one JSON plus one NPZ of
predictions per job and a protocol JSON with code hashes. The cache modules
retain the original server paths under `~/projects/smiles/cache`; moving the
cache requires adapting those modules and regenerating/verifying cache hashes.

From the repository root on the original Slurm server:

```bash
sbatch projects/context_selection/src/fm_transfer/run_transfer_holdout.sbatch
```

Monitor with `squeue -u "$USER"` and the Slurm output file. The original
full Criteo holdout took about 99 minutes per CausalPFN run at K=4096 and
33 minutes per TabPFN-T run at K=4096 on one RTX 2080 Ti; K=1024 has not
been measured in the supplied report. Plan for roughly 6--12 GPU hours for
12 runs, with a 24-hour job limit for queue/throughput variation. This is an
estimate, not a guaranteed runtime. Cluster cache and storage capacity must
be checked before submission.

After the 12 jobs, compare with the original Do-PFN K=1024 full-holdout
outputs. This fails if any run or prediction file is missing, or if the
holdout/evaluation hashes differ:

```bash
"$HOME/ml-env/bin/python" projects/context_selection/src/fm_transfer/summarize_transfer.py \
  --new-output "$HOME/projects/smiles/results/transfer_holdout_k1024" \
  --dopfn-output "$HOME/projects/smiles/results/fm_holdout"
```

The generated `matched_transfer_deltas.csv` has one row per model and seed.
The summary gives descriptive paired changes only; it does not report an
independent-test confidence interval for the cross-model interaction.

For a read-only split/context check on a GPU allocation, use the same Python
environment and paths with `--check` instead of `--run`. Do not run full
holdout inference on a login node.
