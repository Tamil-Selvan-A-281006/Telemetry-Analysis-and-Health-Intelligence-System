# TAAHIS — Experimental Index

## Objective

Evaluate statistical and machine-learning-based telemetry anomaly detection methods using controlled synthetic datasets.

## Completed Experiments

1. Modified Z-score magnitude sensitivity.
2. Statistical vs Isolation Forest comparison.
3. Isolation Forest contamination sensitivity.
4. Random-state stability.
5. Dataset robustness.
6. Anomaly prevalence.
7. Concept drift.
8. Statistical confidence analysis.

## Statistical Validation

* 30 trials per drift condition.
* Five drift conditions.
* Paired detector comparisons.
* 95% confidence intervals.
* 10,000 bootstrap iterations.
* Wilcoxon signed-rank tests.
* Holm-Bonferroni correction.

## Main Findings

* Modified Z-score achieved a lower average false-positive rate than Isolation Forest across the five tested drift conditions.
* The differences remained statistically significant after Holm correction.
* Both detectors experienced increased false-positive rates under stronger concept drift.
* Severe drift substantially degraded both detectors.

## Limitations

* All datasets are synthetic.
* Ground-truth labels are assigned through experiment design.
* Results do not establish real spacecraft telemetry performance.
* Results do not establish universal superiority of either detector.
* Online drift adaptation is not implemented.
* Thresholds require domain validation.

## Reproducibility

Run from the project root:

```powershell
python -m experiments.evaluate_statistical_confidence
python -m pytest -q
```

## Validation

Latest automated test result: 41 passed.
