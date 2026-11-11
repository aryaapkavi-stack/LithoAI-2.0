# LithoAI 2.0
### Evaluating machine learning for semiconductor manufacturing defect prediction

LithoAI 2.0 is an independent experimental research project investigating machine-learning methods for identifying defective observations in semiconductor manufacturing data. It uses the public **UCI SECOM** dataset to study class imbalance, incomplete sensor measurements, timestamp-grouped validation and temporal generalisation.

The project focuses on evaluation methodology and operational trade-offs rather than claiming a production-ready defect predictor. The SECOM dataset is a general semiconductor manufacturing dataset; the experiments do **not** establish lithography-specific performance.

## Research questions

- How do linear and tree-based classifiers rank rare failure observations under timestamp-grouped validation?
- How do recall, precision and inspection workload change when decision thresholds are selected to target a recall level?
- How sensitive are predictions to additional missing sensor readings at inference time?
- How well do models generalise to later chronological periods?
- What sensor-distribution and missingness changes are visible across those periods?

## Dataset

The project uses the UCI SECOM dataset.

| Characteristic | Value |
|---|---:|
| Observations | 1,567 |
| Sensor features | 590 |
| Failure observations | 104 |
| Passing observations | 1,463 |
| Failure prevalence | 6.64% |
| Unique timestamps | 1,534 |
| Features with missing values | 538 |

The data are highly imbalanced and contain substantial missing sensor readings. Observations with the same timestamp are kept together during grouped validation.

Dataset source: [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom).

## Methodology

The primary evaluation includes:

1. Data integrity, missingness and feature-variability analysis.
2. Five-fold `StratifiedGroupKFold` comparison using timestamp groups.
3. Average precision (AP; referred to as PR-AUC in the notebook) for imbalanced classification.
4. Nested grouped threshold selection (five outer folds, four inner folds) with an illustrative 80% recall target.
5. Test-time missingness simulation at 0%, 5%, 10%, 20% and 30%, repeated five times.
6. Expanding-window chronological evaluation on Periods 3–5.
7. Sensor-level distribution and missingness audits, grouped permutation importance and a missingness-indicator comparison.

## Selected results from the primary evaluation

### Grouped model comparison

Pooled out-of-fold average precision:

| Model | Pooled OOF AP |
|---|---:|
| Logistic Regression | 0.1390 |
| Random Forest | 0.1860 |
| HistGradientBoosting | 0.1916 |

The overall failure-prevalence baseline is 0.0664. The separate mean of the five fold-level AP scores was 0.1568 ± 0.0290 for Logistic Regression, 0.2279 ± 0.0973 for Random Forest and 0.2289 ± 0.0997 for HistGradientBoosting. **Pooled AP and mean fold AP are different summaries and should not be interchanged.**

*Configuration note:* the notebook also retains an earlier exploratory grouped benchmark using different model settings. That separate configuration produced pooled AP of 0.1926 (Random Forest) and 0.1880 (HistGradientBoosting). Those values are not the headline results above and should not be mixed with the primary evaluation.

### Nested threshold evaluation

Thresholds were selected from inner-fold out-of-fold predictions and evaluated on untouched outer folds. The 80% recall target is a selection objective, not a guarantee on held-out observations.

| Model | Recall | Precision | Inspection rate |
|---|---:|---:|---:|
| Random Forest | 76.92% | 11.19% | 45.63% |
| HistGradientBoosting | 82.69% | 9.46% | 58.01% |

HistGradientBoosting detected six more failures in the pooled outer folds, but required 194 additional inspections. Neither model dominates across recall, precision and inspection workload.

### Additional test-time missingness

The models were trained on the original training data; artificial missing values were introduced only in test folds. Results below compare 0% with 30% additional missingness.

| Model | AP (0% → 30%) | Recall (0% → 30%) | Inspection rate (0% → 30%) |
|---|---:|---:|---:|
| HistGradientBoosting | 0.2289 → 0.1994 | 82.82% → 75.24% | 58.03% → 50.47% |
| Random Forest | 0.2279 → 0.2072 | 77.45% → 87.96% | 45.66% → 64.38% |

Random Forest's increase in recall came with a larger inspection burden. The models therefore responded differently to additional missingness; recall alone is not sufficient to describe robustness.

### Expanding-window temporal evaluation

| Test period | Test observations | Failures | Failure prevalence | Random Forest AP | HistGradientBoosting AP |
|---|---:|---:|---:|---:|---:|
| 3 | 313 | 11 | 0.0351 | 0.2795 | 0.2979 |
| 4 | 309 | 11 | 0.0356 | 0.0579 | 0.0471 |
| 5 | 316 | 17 | 0.0538 | 0.0930 | 0.0543 |

Mean AP across these three periods was 0.1435 for Random Forest and 0.1331 for HistGradientBoosting; mean period-specific prevalence was 0.0415. The later periods had only 11–17 failures each, so their estimates are sensitive to a small number of observations. The results do not establish that sensor drift caused the decline.

### Sensor and feature analysis

The robust sensor audit found 1, 7 and 9 sensors with absolute median shifts above one training IQR in Periods 3, 4 and 5, respectively. Sensors with missingness changes above 10 percentage points numbered 36, 44 and 56. Grouped permutation importance ranked Sensor 59 highest by mean importance (0.003459), but it appeared in the top 20 in only three of five folds; these values are exploratory and not causal evidence.

Adding missingness indicators did not consistently improve chronological AP: the with-minus-without changes were -0.0306, +0.0028 and -0.0035 in Periods 3, 4 and 5.

## Limitations

- The dataset contains only 104 failure observations, and each chronological test period contains relatively few failures.
- The study uses one public dataset; external manufacturing-line generalisation has not been tested.
- SECOM does not establish that its measurements correspond specifically to a lithography process step.
- Artificial test-time masking may not reproduce real industrial missing-data mechanisms.
- Sensor importance and distribution changes are descriptive, not causal.
- There has been no production deployment, prospective validation or operational cost evaluation.

## Repository contents

- **Primary notebook:** [`notebooks/LithoAI_2_0_Research_Notebook.ipynb`](notebooks/LithoAI_2_0_Research_Notebook.ipynb)
- **Detailed experimental tables:** [`notebooks/results/README.md`](notebooks/results/README.md)
- **Methodology and reproducibility notes:** [`notebooks/documentation/README.md`](notebooks/documentation/README.md)
- **Python dependencies:** [`requirements.txt`](requirements.txt)

The older `lithoai_experiments.ipynb` is an earlier exploratory notebook. Use the LithoAI 2.0 research notebook as the current reference.

## Reproducibility

The notebook downloads the UCI SECOM dataset at runtime. To reproduce the experiments, use Python with the packages in `requirements.txt`, an internet connection, and run the notebook from the first cell to the last in a fresh environment. Confirm that the outputs complete and match the reported tables before reusing the numbers.

## Author

**Aryaa Kavi**  
B.Tech — VLSI Design and Technology
