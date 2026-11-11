# LithoAI 2.0 — Experimental Results

This page records the primary experiment outputs from `LithoAI_2_0_Research_Notebook.ipynb`. It distinguishes pooled out-of-fold average precision from the mean of fold-specific AP scores. These are different aggregations.

## 1. Dataset summary

| Quantity | Value |
|---|---:|
| Observations | 1,567 |
| Sensor features | 590 |
| Failures | 104 |
| Passing observations | 1,463 |
| Failure prevalence | 0.0664 |
| Unique timestamps | 1,534 |
| Features with missing values | 538 |
| Constant/single-value sensors | 116 |

Three timestamp groups contain both labels. Keeping timestamp groups intact prevents records sharing a timestamp from being split across train and validation folds; it does not resolve the contradictory labels within those groups.

## 2. Primary grouped model comparison

Primary pipeline: Logistic Regression, Random Forest (300 trees) and HistGradientBoosting (200 iterations; class-balanced), evaluated using five-fold `StratifiedGroupKFold` with timestamp groups (`random_state=42`).

| Model | Pooled OOF AP | Mean fold AP ± SD |
|---|---:|---:|
| Logistic Regression | 0.1390 | 0.1568 ± 0.0290 |
| Random Forest | 0.1860 | 0.2279 ± 0.0973 |
| HistGradientBoosting | 0.1916 | 0.2289 ± 0.0997 |

The overall failure prevalence was 0.0664. Pooled AP is calculated once using all out-of-fold predictions; mean fold AP averages the five fold-specific AP values. Their difference is expected because AP is ranking-based and score distributions from separate folds are not necessarily on a common scale.

### Separate exploratory configuration

An earlier grouped benchmark in the notebook uses different estimator settings (including a 500-tree Random Forest and 150-iteration HistGradientBoosting). It produced pooled AP of 0.1926 for Random Forest and 0.1880 for HistGradientBoosting. These results are not directly interchangeable with the primary model comparison above.

## 3. Nested threshold evaluation

Five outer folds and four inner folds were used. The inner folds selected a threshold to target at least 80% recall; the threshold was then evaluated on the untouched outer fold.

| Model | TP | FP | TN | FN | Recall | Precision | Inspection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 80 | 635 | 828 | 24 | 0.7692 | 0.1119 | 0.4563 |
| HistGradientBoosting | 86 | 823 | 640 | 18 | 0.8269 | 0.0946 | 0.5801 |

The target recall was not consistently met on every outer fold. Pooled recall reflects the combined outer-fold predictions. HistGradientBoosting detected six more failures but triggered 194 more inspections than Random Forest.

## 4. Additional test-time missingness

Five masking repetitions were performed at 0%, 5%, 10%, 20% and 30% additional missingness. Masking was applied only to originally observed values in test data; training data were unchanged.

| Model | Added missingness | Mean AP | Mean recall | Mean inspection rate |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0% | 0.2289 | 0.8282 | 0.5803 |
| HistGradientBoosting | 5% | 0.2315 | 0.8209 | 0.5676 |
| HistGradientBoosting | 10% | 0.2304 | 0.8120 | 0.5582 |
| HistGradientBoosting | 20% | 0.2115 | 0.7985 | 0.5362 |
| HistGradientBoosting | 30% | 0.1994 | 0.7524 | 0.5047 |
| Random Forest | 0% | 0.2279 | 0.7745 | 0.4566 |
| Random Forest | 5% | 0.2318 | 0.7758 | 0.4780 |
| Random Forest | 10% | 0.2283 | 0.7905 | 0.5057 |
| Random Forest | 20% | 0.2211 | 0.8501 | 0.5683 |
| Random Forest | 30% | 0.2072 | 0.8796 | 0.6438 |

The Random Forest recall increase at higher missingness coincided with a higher inspection rate and more false positives. Small non-monotonic AP changes occur at intermediate levels.

## 5. Expanding-window temporal evaluation

Unique timestamps were divided into five chronological blocks. The models were trained on all earlier blocks and tested on Periods 3–5.

| Test period | Train rows | Train failures | Test rows | Test failures | Baseline AP | RF AP | HGB AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 629 | 65 | 313 | 11 | 0.0351 | 0.2795 | 0.2979 |
| 4 | 942 | 76 | 309 | 11 | 0.0356 | 0.0579 | 0.0471 |
| 5 | 1,251 | 87 | 316 | 17 | 0.0538 | 0.0930 | 0.0543 |

Mean AP across periods: Random Forest 0.1435; HistGradientBoosting 0.1331. Mean period-specific baseline AP: 0.0415. Period-specific positive counts are small, and the experiment cannot identify the cause of performance variation.

## 6. Sensor distribution and missingness audit

| Test period | Sensors analysed | Sensors all-missing in training | Shift >1 training IQR | Shift >2 training IQR | Sensors with missingness change >10 pp | Mean absolute median shift (IQR units) |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 444 | 16 | 1 | 0 | 36 | 0.1787 |
| 4 | 460 | 0 | 7 | 1 | 44 | 0.2213 |
| 5 | 464 | 0 | 9 | 5 | 56 | 0.2842 |

Sensors with no observed training values or no valid positive training IQR were excluded from the standardised median-shift calculation. Sensor 72's median and missingness both changed across periods; Sensor 545 showed a higher median and wider spread in Period 5. These are descriptive shifts and are not evidence of causation.

## 7. Grouped permutation importance

Five timestamp-grouped folds were used, with 10 permutation repeats per fold and average precision as the scoring metric. Mean fold AP was 0.2368 ± 0.0963 for this feature-importance experiment. Sensor 59 had the largest mean importance (0.003459; SD 0.013925) and appeared in the top 20 in three of five folds. Correlated features can share importance; these values are not causal explanations.

## 8. Missingness-indicator comparison

Random Forest chronological AP with minus without missingness indicators:

| Test period | Failures | AP without indicators | AP with indicators | Difference |
|---|---:|---:|---:|---:|
| 3 | 11 | 0.2795 | 0.2489 | -0.0306 |
| 4 | 11 | 0.0579 | 0.0607 | +0.0028 |
| 5 | 17 | 0.0930 | 0.0895 | -0.0035 |

There was no consistent improvement across the three periods.

## Interpretation and limitations

The primary experiments show modest AP above the overall failure-prevalence baseline, substantial threshold/inspection trade-offs, sensitivity to test-time missingness, and variable performance across later chronological periods. The dataset contains only 104 failures and the chronological test periods contain 11–17 failures each. Only one public dataset was studied; SECOM does not establish lithography-specific performance. Sensor shifts and permutation importance are exploratory, not causal. No production deployment or external manufacturing-line validation was performed.
