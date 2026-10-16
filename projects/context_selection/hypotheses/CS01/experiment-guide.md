# Context selection across the five uplift datasets

`multi_dataset_context.py` is an independent screening runner. It does not
rewrite the completed Hillstrom pilot. The default dataset set is Hillstrom,
Retailhero-uplift, LZD, Criteo-ITE-v2.1, and Orange Telecom Churn.

## What the run tests

For the same training bank, evaluation rows, CausalPFN checkpoint, and context
budget `K`, it compares:

- One **shared random context** for all queries, repeated with context seeds
  42, 43, and 44 by default;
- A separate random context for each query, with those same seeds;
- Feature-space nearest neighbours per treatment arm;
- Weak-learner-effect nearest neighbours per treatment arm. By default, each
  train-bank score is **out-of-fold**: the weak model scoring that row did not
  train on its outcome. Eval scores are the mean of the same five inner-fold
  models. Preprocessing is fitted separately on each inner training fold;
- The weak learner by itself as a non-PFN anchor.

Every PFN context has `K/2` control and `K/2` treated examples. Random seeds
within a fold measure sensitivity to context construction. They are averaged
inside that fold, not treated as independent train/test splits. `--folds 5`
uses five outer test folds; training banks overlap, so even those five estimates
are not independent datasets. Contexts and predictions are saved per fold.

## Dataset decisions and limits

| Dataset | Benefit outcome | Features | Split |
| --- | --- | --- | --- |
| Hillstrom | `visit` | Eight pilot covariates | Stratified |
| Retailhero-uplift | `Y` | Gender indicators, age, first issue time | Stratified |
| LZD | `Y` | Cleaned `f*` fields | Stratified group split by feature-vector hash |
| Criteo-ITE-v2.1 | `visit` | Cleaned `f*` fields | Uniform 100k-row sample, then stratified group split |
| Orange Telecom Churn | `1 - churn` | Numeric PC coordinates; FACTOR labels omitted | Stratified |

The Criteo sample limit is a deliberate memory/compute limit, not an evaluation
of all 13.98 million rows. The same sampled population and capped train bank are
used for all methods in that run. The sample size and source-file hashes are
recorded. `--criteo-sample-seed` controls this source sample separately from
`--split-seed`. The group split for LZD and Criteo prevents an identical feature
vector appearing on both sides. Preprocessing and any dimension reduction are
fitted on the train bank only; the effect-score cross-fit repeats preprocessing
inside each inner fold. The `weak` anchor is still fitted on the full train bank
and evaluated out of sample, so it is not the mean of the inner-fold models.
Orange's benefit coding and assignment mechanism must be independently
confirmed before article claims. Its old weak learner ignored treatment and
returned zero effect for every eval row, making old effect retrieval degenerate.

The runner uses `--train-max 51200`, `--eval-max 4096`, `K=512`, one split,
three context seeds, and CPU by default. Orange has fewer than 4096 holdout
rows, so it uses the full available holdout. `K=2048` is a separate budget
sensitivity check; `K=4096` is not required for the first cross-dataset pass.

## Commands

Run from the repository root with its prepared environment. Local data are read from `artifacts/local/data_A_cleaned/`, and new full outputs are written under `artifacts/local/multi_dataset_context_results/`:

```bash
.venv/bin/python -u projects/context_selection/src/multi_dataset_context.py --audit
```

This validates all five data adapters, IDs, treatment/outcome coding, and
splits without loading CausalPFN. Without `--audit` or `--run`, the script
checks each selector and two-query model inference, but produces no outcome
metrics. Pin the same checkpoint used in the Hillstrom pilot:

```bash
.venv/bin/python -u projects/context_selection/src/multi_dataset_context.py \
  --revision 83aad07da1cb077cfda4236878a1b07dc9f72a54
```

One-split, all-dataset pilot with both random controls and corrected
out-of-fold effect scores:

```bash
.venv/bin/python -u projects/context_selection/src/multi_dataset_context.py --run \
  --device cpu --context-sizes 512 --context-seeds 42 43 44 \
  --revision 83aad07da1cb077cfda4236878a1b07dc9f72a54
```

To check the new selector on the same split as the previous pilot, run the
above command with `--datasets Hillstrom Retailhero-uplift LZD Criteo-ITE-v2.1`.
It remains **screening**, not a split-replicated confirmation. For the next
multi-split run, use `--folds 5 --context-sizes 512` on these four datasets;
keep Orange as a separate sensitivity analysis until its data semantics and
degenerate weak model are resolved. The previous in-sample selector can still
be reproduced with `--effect-score-mode legacy`, but it must be labeled as an
ablation, not silently mixed with the corrected results. Cross-fitting adds
five weak-model fits per outer fold and the 5-fold PFN run is substantially
more expensive than the one-split screening pass. K=1024/2048 comes later.

Suggested staged commands after the small smoke checks:

```bash
# Diagnostic rerun of the previous single split with corrected effect scores.
.venv/bin/python -u projects/context_selection/src/multi_dataset_context.py --run \
  --datasets Hillstrom Retailhero-uplift LZD Criteo-ITE-v2.1 \
  --split-seed 42 --criteo-sample-seed 42 \
  --context-sizes 512 --context-seeds 42 43 44 \
  --effect-score-mode crossfit --nuisance-folds 5 --nuisance-seed 42 \
  --revision 83aad07da1cb077cfda4236878a1b07dc9f72a54

# Main multi-split series (substantially longer; no K sweep yet).
.venv/bin/python -u projects/context_selection/src/multi_dataset_context.py --run \
  --datasets Hillstrom Retailhero-uplift LZD Criteo-ITE-v2.1 \
  --folds 5 --split-seed 42 --criteo-sample-seed 42 \
  --context-sizes 512 --context-seeds 42 43 44 \
  --effect-score-mode crossfit --nuisance-folds 5 --nuisance-seed 42 \
  --revision 83aad07da1cb077cfda4236878a1b07dc9f72a54
```

The diagnostic rerun still has only one outer split; the main series has five
folds on one fixed Criteo source sample. To study source-sample variability,
repeat Criteo with a separately specified `--criteo-sample-seed`. Every run
uses a fresh timestamped output directory, and incomplete runs are not
automatically resumed. The previous one-split run spent roughly 18 minutes of
PFN inference per dataset; five folds on four datasets therefore imply several
hours of PFN inference before the additional cross-fit costs.

## Outputs and interpretation

The timestamped root contains `protocol.json`, `model.json`, `failures.json`,
`all_metrics.csv`, `all_paired_deltas.csv`, `summary.csv`,
`selector_effects.csv`, `selector_effects_summary.csv`, and a selector-effect
plot for every requested K. Each dataset/fold also has its source-row IDs,
configuration, contexts, predictions, timings, metrics, and paired deltas.
Each fold additionally saves `effect_scores_train.parquet` with its inner-fold
assignment and `effect_scores_eval.parquet` for checking score distributions.
The exact effect-score mode, inner-fold count/seed and Criteo sample seed are
recorded in the protocol.

The main research table is `selector_effects_summary.csv`: the mean and range
of **selector minus random** Qini differences across outer folds, plus the
number of folds with a positive difference. `all_paired_deltas.csv` has
within-fold paired bootstrap intervals conditional on that fold's fitted
model and selected contexts. These intervals do not account for retraining,
dataset selection, or multiple comparisons. Uplift at 5/10/20/30% can be
undefined if a selected top group contains only treated or only control users;
such cases are saved as missing rather than filled with zero.

An all-dataset smoke check verifies that the pipeline can execute. It is not
evidence that a selector improves targeting. Report per-dataset estimates and
uncertainty even if some effects are negative; do not select only the datasets
or K values that favor the hypothesis.
