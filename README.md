# Hype Check: Challenging Causal Foundation Models for Uplift

**One-sentence summary:** This project tests whether strong CATE estimation accuracy from a ready-to-use causal foundation model, CausalPFN, translates into better uplift ranking and budget-constrained targeting.

## Abstract

Causal foundation models such as CausalPFN aim to provide ready-to-use treatment-effect estimates without task-specific model tuning. This project evaluates whether that promise is sufficient for uplift modeling, where the operational goal is often not to estimate every individual effect accurately, but to prioritize a limited intervention budget.

The current work contains two preliminary experiments. First, a synthetic randomized proof of concept with known true CATE verifies the implementation and confirms that PEHE, regret, and top-20% targeting value can be computed correctly. Second, a pilot evaluation on the Hillstrom randomized email campaign compares CausalPFN against S-Learner and T-Learner baselines using out-of-fold uplift scores, Qini AUC, AUUC, and Uplift@20.

The results should be interpreted conservatively. In the synthetic setting, CausalPFN is both the best CATE estimator and the best top-20% targeter. In the Hillstrom pilot, S-Learner has the highest point estimate for Qini AUC and AUUC, while CausalPFN has the highest point estimate for Uplift@20. The CausalPFN versus S-Learner Uplift@20 gap is only 0.2335 percentage points and its paired bootstrap confidence interval crosses zero. The current evidence validates the evaluation pipeline and suggests that model choice may depend on the targeting objective, but it does not confirm the main hypothesis.

## Motivation

Treatment-effect models are often evaluated by how accurately they estimate conditional effects. In applied marketing, healthcare, and policy settings, however, the decision problem is usually a ranking problem: who should receive the intervention when only a fraction of the population can be targeted?

This distinction matters because an estimator can reduce average CATE error while still ranking the top actionable segment poorly. Conversely, a model with imperfect CATE calibration may still produce useful treatment prioritization. Recent work on real versus semi-simulated evaluation argues that counterfactual metrics and observable real-data metrics can prefer different methods (Panagopoulos, 2026). This project uses that concern as the central evaluation stress test for CausalPFN.

## Research Question and Hypothesis

**Research question:** Does lower CATE estimation error necessarily imply better uplift ranking and better fixed-budget targeting?

**Hypothesis:** Lower PEHE does not necessarily guarantee higher Qini, AUUC, or Uplift@k. The best treatment-effect estimator may not be the best targeter under a fixed intervention budget.

**Current status:** The hypothesis is not confirmed. The synthetic experiment does not show a disagreement between estimation and targeting rankings, and PEHE cannot be computed on the real Hillstrom campaign because individual ground-truth CATE values are unavailable.

## Background

### Potential Outcomes

For each unit with features \(X=x\), let \(Y(1)\) be the potential outcome under treatment and \(Y(0)\) the potential outcome under control. The observed outcome is

\[
Y = T Y(1) + (1 - T)Y(0),
\]

where \(T \in \{0, 1\}\) is treatment assignment. The fundamental difficulty is that only one potential outcome is observed for each unit.

### CATE

The conditional average treatment effect is

\[
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x].
\]

CATE estimation asks for accurate individual-level effect estimates, usually evaluated against known or simulated ground truth when both potential outcomes can be recovered or constructed.

### Uplift Modeling

Uplift modeling uses treatment-effect scores to rank units by expected incremental response. In a campaign setting, the relevant question is often: if only the top \(k\%\) can be targeted, how much incremental outcome is achieved?

### Effect Estimation vs. Treatment Prioritization

CATE estimation and treatment prioritization are related but not identical. PEHE penalizes pointwise estimation error across the population. Qini, AUUC, and Uplift@k evaluate ranking and policy value under randomized assignment. A model can be well calibrated on average while making ranking mistakes near the intervention threshold.

## Why PEHE May Disagree With Targeting Metrics

PEHE averages squared CATE error over all evaluated units:

\[
\text{PEHE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{\tau}(x_i)-\tau(x_i))^2}.
\]

Targeting metrics depend more strongly on ordering, especially near the selected budget cutoff. Disagreement can occur because:

- PEHE is sensitive to global pointwise error, while targeting depends on rank order.
- Errors among non-targeted units may dominate PEHE but have little policy impact.
- Small score perturbations around the top-k threshold can change the targeted set.
- Real randomized campaigns provide observable uplift metrics but not true individual CATE.
- Low-conversion regimes can produce weak and noisy ranking signals.

This is consistent with the evaluation gap discussed by Panagopoulos (2026), the real-world benchmark concerns raised by Yu and Sun (2024), and the low-conversion motivation in Fuchs and Kreiss (2026).

## Experimental Design

The project is split into two completed stages and a planned robustness stage.

| Stage | Purpose | Ground-truth CATE available? | Main metrics | Status |
|---|---|---:|---|---|
| Synthetic proof of concept | Validate implementation and metric computation | Yes | PEHE, Value@20, Regret@20 | Completed pilot |
| Hillstrom real-RCT pilot | Evaluate real randomized campaign targeting | No | Qini AUC, AUUC, Uplift@20 | Completed pilot |
| Robustness study | Test stability across seeds, folds, datasets, and baselines | Mixed | PEHE, Qini, AUUC, Uplift@k, rank agreement | Planned |

## Datasets

### Synthetic Proof of Concept

The synthetic experiment uses a balanced randomized dataset with known true CATE. Treatment is assigned randomly with probability 0.5. The outcome is generated from a known baseline response surface plus a heterogeneous treatment effect. Because \(\tau(x)\) is known, PEHE and oracle targeting regret can be computed directly.

| Property | Value |
|---|---:|
| Sample size | 5,000 |
| Feature dimension | 5 |
| Test fraction | 30% |
| Treatment assignment | Randomized, \(p=0.5\) |
| Outcome type | Continuous synthetic outcome |
| True CATE | Known by construction |

### Hillstrom Randomized Campaign

The real-data pilot uses the Hillstrom email campaign dataset loaded through `sklift.datasets.fetch_hillstrom`. The official scikit-uplift documentation describes the dataset as a 64,000-customer randomized email test with treatment segments for men's email, women's email, and no email.

This project currently evaluates one binary treatment contrast:

- **Treatment:** Women's Email
- **Control:** No Email
- **Outcome:** website visit

| Quantity | Value |
|---|---:|
| Selected contrast size | 42,693 customers |
| Treatment share | approximately 0.501 |
| Overall visit rate | approximately 0.129 |
| Control visit rate | 10.62% |
| Treatment visit rate | 15.14% |
| Observed average treatment effect | approximately 4.52 percentage points |

Pilot protocol:

| Setting | Value |
|---|---:|
| Stratified pilot subsample | 12,000 observations |
| Evaluation split | 2-fold honest splitting |
| Random seed | 42 |
| Prediction type | Out-of-fold CATE/uplift scores |
| CausalPFN context budget | 512 |
| CausalPFN maximum query size | 256 |
| CausalPFN retrieved neighbours | 128 |
| Bootstrap iterations | 1,000 |
| Bootstrap scope | Conditional on fixed out-of-fold predictions |

## Compared Models

### CausalPFN

CausalPFN is a transformer-based causal effect estimator using in-context learning and amortized inference (Balazadeh et al., 2025). The official repository describes it as a ready-to-use library for causal effect estimation with CATE and uplift-modeling examples. In this project, it is used through `CATEEstimator` without project-specific training.

### S-Learner

The S-Learner fits one predictive model for the outcome with treatment included as an additional feature. The CATE score is computed as the predicted treated outcome minus the predicted control outcome.

### T-Learner

The T-Learner fits separate outcome models for treated and control units. The CATE score is the difference between the two predicted potential outcomes.

### Planned Baselines

The following baselines are planned and should not be read as completed experiments: X-Learner, R-Learner, DR-Learner, Causal Forest, uplift random forest, and tuned shared predictive backbones.

## Evaluation Metrics

### CATE

\[
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x].
\]

Requires causal identification assumptions for estimation. Ground-truth individual CATE is only directly available in synthetic or semi-synthetic data.

### PEHE

\[
\text{PEHE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{\tau}(x_i)-\tau(x_i))^2}.
\]

Requires true individual CATE, so it is reported only for the synthetic experiment.

### Value@20

Value@20 is the average true CATE among the top 20% of units ranked by predicted treatment effect:

\[
\text{Value@20} = \frac{1}{|S_{20}|}\sum_{i \in S_{20}} \tau(x_i).
\]

It requires true CATE and is therefore used only in the synthetic experiment.

### Regret@20

\[
\text{Regret@20} = \text{Oracle Value@20} - \text{Model Value@20}.
\]

It requires true CATE and an oracle top-20% set, so it is used only in the synthetic experiment.

### Qini

Qini evaluates cumulative incremental outcomes as units are targeted in descending order of predicted uplift. It is observable on randomized experiments because treatment assignment supplies unbiased treated-control comparisons along the ranking.

### AUUC

AUUC is the area under the uplift curve. Like Qini, it evaluates ranking quality on randomized real data without requiring individual ground-truth CATE.

### Uplift@20

Uplift@20 estimates the incremental response rate among the top 20% of units ranked by model score. It is a fixed-budget targeting metric and does not require true individual CATE on randomized data.

## Synthetic Results

The synthetic values below come from the executed notebook output record. No separate synthetic results CSV is currently committed.

| Model | PEHE (lower is better) | Value@20 (higher is better) | Oracle Value@20 | Regret@20 (lower is better) | Relative regret |
|---|---:|---:|---:|---:|---:|
| CausalPFN | 0.0631 | 1.0548 | 1.0582 | 0.0034 | approximately 0.32% |
| T-Learner | 0.2015 | 1.0316 | 1.0582 | 0.0266 | not separately reported |
| S-Learner | 0.2082 | 1.0461 | 1.0582 | 0.0121 | not separately reported |

**Interpretation:** CausalPFN is both the most accurate CATE estimator and the best top-20% targeter in this simple balanced synthetic setting. This validates the implementation and metric pipeline, but it does not demonstrate a disagreement between estimation and targeting rankings.

## Hillstrom Pilot Results

Saved result tables:

- [metrics/hillstrom_pilot_metrics_ci.csv](metrics/hillstrom_pilot_metrics_ci.csv)
- [metrics/hillstrom_pilot_comparisons.csv](metrics/hillstrom_pilot_comparisons.csv)
- [metrics/hillstrom_pilot_configuration.csv](metrics/hillstrom_pilot_configuration.csv)

No figure file is currently present in the repository. The notebook code references `hillstrom_pilot_results.png`, but that file is not committed.

### Point Estimates and 95% Bootstrap Intervals

| Model | Qini AUC [95% CI] (higher is better) | AUUC [95% CI] (higher is better) | Uplift@20, pp [95% CI] (higher is better) | Qini rank |
|---|---:|---:|---:|---:|
| S-Learner | 0.0289 [-0.0003; 0.0576] | 0.0137 [0.0000; 0.0274] | 5.25 [2.62; 8.10] | 1 |
| CausalPFN | 0.0034 [-0.0271; 0.0343] | 0.0019 [-0.0121; 0.0165] | 5.48 [2.66; 8.50] | 2 |
| T-Learner | 0.0006 [-0.0282; 0.0303] | 0.0000 [-0.0131; 0.0136] | 4.34 [1.44; 7.07] | 3 |

### Paired Bootstrap Comparisons

| Comparison | Metric | Difference | 95% CI | Interpretation |
|---|---|---:|---:|---|
| S-Learner minus CausalPFN | Qini AUC | 0.0254 | [0.0011; 0.0514] | S-Learner is higher in this conditional paired bootstrap |
| S-Learner minus CausalPFN | AUUC | 0.0117 | [0.0005; 0.0238] | S-Learner is higher in this conditional paired bootstrap |
| CausalPFN minus S-Learner | Uplift@20, pp | 0.2335 | [-2.2497; 2.7332] | No statistically clear difference |

**Interpretation:** S-Learner has the highest point estimate of overall ranking quality according to Qini and AUUC. CausalPFN has the highest point estimate of Uplift@20, but the difference from S-Learner is only 0.2335 percentage points and is not statistically clear under the paired bootstrap interval. The low absolute Qini values indicate weak or noisy ranking signal.

## Interpretation

The current Hillstrom pilot separates two evaluation objectives:

- **Overall ranking quality:** S-Learner is favored by Qini AUC and AUUC point estimates, and by the current conditional paired bootstrap comparison against CausalPFN.
- **Fixed-budget targeting at 20%:** CausalPFN has the highest point estimate, but the CausalPFN-S-Learner difference is small and the confidence interval includes zero.

This is a useful preliminary result because it shows how different decision metrics can point to different models. It is not yet evidence that CausalPFN is superior, and it is not a confirmation that the best PEHE model differs from the best targeter.

## Limitations

- Only one real dataset has been evaluated.
- Only one treatment contrast and one outcome are used.
- The Hillstrom pilot uses a 12k subsample instead of the complete selected contrast.
- The current out-of-fold protocol uses two folds instead of five.
- Only one random seed has been evaluated.
- CausalPFN is run with a constrained context budget.
- The S-Learner and T-Learner baselines are untuned or minimally tuned.
- X-Learner, R-Learner, DR-Learner, and Causal Forest are not implemented yet.
- PEHE is not available on the real Hillstrom dataset.
- Bootstrap uncertainty is conditional on fixed out-of-fold predictions and does not capture training-seed, fold-selection, or model-refitting variability.
- Absolute Qini scores are weak, suggesting a noisy or limited ranking signal.
- The main hypothesis is not confirmed.

## Reproducibility

Primary notebook:

- [SMILES_HypeCheck.ipynb](SMILES_HypeCheck.ipynb)

The notebook installation cell currently installs:

```python
%pip install -q causalpfn scikit-uplift
```

Main imported packages:

- `causalpfn`
- `scikit-uplift`
- `torch`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`

There is no committed `requirements.txt`, `pyproject.toml`, or environment lockfile in the current repository. Reproducibility therefore depends on the notebook install cell and the package versions resolved at run time.

### Repository Audit Notes

- The checked notebook currently has no stored cell outputs.
- Hillstrom confidence intervals and paired comparisons are available in committed CSV files under `metrics/`.
- The notebook code saves result CSVs to the repository root, while the committed CSVs are stored under `metrics/`.
- The notebook code references `hillstrom_pilot_results.png`, but no PNG file is currently present in the repository.

## Repository Structure

```text
.
+-- README.md
+-- SMILES_HypeCheck.ipynb
+-- metrics
    +-- hillstrom_pilot_comparisons.csv
    +-- hillstrom_pilot_configuration.csv
    +-- hillstrom_pilot_metrics_ci.csv
```

## How to Run the Notebook

1. Open [SMILES_HypeCheck.ipynb](SMILES_HypeCheck.ipynb) in Jupyter or VS Code.
2. Run the installation cell if the environment does not already contain `causalpfn` and `scikit-uplift`.
3. Run cells from top to bottom.
4. Recompute the synthetic proof of concept.
5. Recompute the Hillstrom out-of-fold predictions.
6. Recompute the bootstrap confidence intervals.
7. Compare regenerated CSV outputs against the committed files in [metrics](metrics).

The full notebook may require network access for package installation, dataset download, and CausalPFN model loading.

## Planned Experiments

- Five-fold Hillstrom evaluation.
- Multiple random seeds.
- Full selected Hillstrom contrast instead of a 12k pilot subsample.
- Hillstrom Men's Email versus No Email contrast.
- Low-conversion outcome analysis using conversion.
- IHDP and ACIC 2016 experiments with known or semi-synthetic CATE.
- X-Learner, R-Learner, and DR-Learner baselines.
- Causal Forest baseline.
- Tuned common predictive backbones across learners.
- Paired bootstrap confidence intervals for all main model comparisons.
- Model-rank agreement analysis using Kendall's tau.
- PEHE-based model-selection regret.
- Stress tests for CausalPFN context size, control-group imbalance, and low-conversion regimes.

## References

- Balazadeh, V., Kamkari, H., Thomas, V., Li, B., Ma, J., Cresswell, J. C., and Krishnan, R. G. (2025). *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*. arXiv preprint. [https://arxiv.org/abs/2506.07918](https://arxiv.org/abs/2506.07918)
- vdblm. (2025). *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*. Official software repository. [https://github.com/vdblm/CausalPFN](https://github.com/vdblm/CausalPFN)
- Panagopoulos, G. (2026). *Real vs. Semi-Simulated: Rethinking Evaluation for Treatment Effect Estimation*. arXiv preprint. [https://arxiv.org/abs/2605.10430](https://arxiv.org/abs/2605.10430)
- Yu, H., and Sun, Y. (2024; revised 2025). *Do Contemporary Causal Inference Models Capture Real-World Heterogeneity? Findings from a Large-Scale Benchmark*. arXiv preprint. [https://arxiv.org/abs/2410.07021](https://arxiv.org/abs/2410.07021)
- Fuchs, M., and Kreiss, D. (2026). *Beyond Differences: Doubly Robust Meta-Learners for Ratio-Based Treatment Effects*. arXiv preprint. [https://arxiv.org/abs/2605.26288](https://arxiv.org/abs/2605.26288)
- Uehara, E. (2026). *Calibrated Inference for the Conditional Average Treatment Effect in the Few-Placebo Regime via Gaussian Processes*. arXiv preprint. [https://arxiv.org/abs/2605.27473](https://arxiv.org/abs/2605.27473)
- scikit-uplift contributors. (2022). *sklift.datasets.fetch_hillstrom documentation*. Official documentation for the Hillstrom dataset loader. [https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html)

## Scientific Accuracy Checklist

- The project is described as a pilot, not a completed benchmark.
- CausalPFN is not described as superior.
- The main PEHE-versus-targeting hypothesis is explicitly marked as unconfirmed.
- Hillstrom PEHE is not reported because true individual CATE is unavailable.
- Uplift@20 superiority for CausalPFN is described as a point estimate only.
- Bootstrap intervals are described as conditional on fixed out-of-fold predictions.
- Planned baselines and datasets are clearly separated from completed experiments.
