from scipy.stats import wilcoxon
import csv
import numpy as np

from collections import defaultdict
from scipy.stats import t


# 1. Experiment configuration

INPUT_FILE = "data/concept_drift_results.csv"
CONFIDENCE_LEVEL = 0.95

BOOTSTRAP_ITERATIONS = 10000


# 2. Store experimental results

results = defaultdict(lambda: defaultdict(list))


# 3. Load experimental results

with open(INPUT_FILE, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        phase = row["phase"]
        detector = row["detector"]
        trial = int(row["trial"])

        fpr = float(row["false_positive_rate"])

        results[phase][detector].append({
            "trial": trial,
            "fpr": fpr
        })


# 4. Calculate confidence intervals

print("\nTAAHIS — STATISTICAL CONFIDENCE ANALYSIS")

for phase, detectors in results.items():

    print(f"\n--- {phase.upper()} ---")

    for detector, records in detectors.items():

        scores = np.array([
            record["fpr"]
            for record in records
        ])

        count = len(scores)

        if count < 2:
            print(f"Insufficient trials for {detector}")
            continue

        mean = np.mean(scores)
        std = np.std(scores, ddof=1)

        standard_error = std / np.sqrt(count)

        t_critical = t.ppf(
            1 - (1 - CONFIDENCE_LEVEL) / 2,
            df=count - 1
        )

        margin = t_critical * standard_error

        lower = max(0.0, mean - margin)
        upper = min(1.0, mean + margin)

        print(f"\nDetector: {detector}")
        print(f"Trials: {count}")
        print(f"Mean FPR: {mean:.4f}")
        print(f"Standard deviation: {std:.4f}")

        print(
            f"95% Confidence Interval: "
            f"[{lower:.4f}, {upper:.4f}]"
        )


# 5. Paired detector comparison

print("\n--- PAIRED DETECTOR COMPARISON ---")

rng = np.random.default_rng(42)

# Store p-values for multiple-comparison correction
phases = []
p_values = []

for phase, detectors in results.items():

    if (
        "Modified Z-Score" not in detectors
        or "Isolation Forest" not in detectors
    ):
        continue

    statistical = {
        record["trial"]: record["fpr"]
        for record in detectors["Modified Z-Score"]
    }

    isolation = {
        record["trial"]: record["fpr"]
        for record in detectors["Isolation Forest"]
    }

    common_trials = sorted(
        set(statistical) & set(isolation)
    )

    statistical_values = np.array([
        statistical[trial]
        for trial in common_trials
    ])

    isolation_values = np.array([
        isolation[trial]
        for trial in common_trials
    ])

    differences = statistical_values - isolation_values

    print(f"\nPhase: {phase}")
    print(f"Paired trials: {len(differences)}")

    print(
        f"Mean FPR Difference: "
        f"{np.mean(differences):.4f}"
    )

    print(
        f"Difference Standard Deviation: "
        f"{np.std(differences, ddof=1):.4f}"
    )

    # Wilcoxon signed-rank test

    test_result = wilcoxon(
        statistical_values,
        isolation_values,
        alternative="two-sided"
    )

    print(f"Wilcoxon statistic: {test_result.statistic:.4f}")
    print(f"P-value: {test_result.pvalue:.10f}")

    # Store results for Holm correction

    phases.append(phase)
    p_values.append(test_result.pvalue)

    # Bootstrap confidence interval

    bootstrap_means = []

    for _ in range(BOOTSTRAP_ITERATIONS):

        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True
        )

        bootstrap_means.append(np.mean(sample))

    lower_ci, upper_ci = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    print(
        f"95% Bootstrap CI: "
        f"[{lower_ci:.4f}, {upper_ci:.4f}]"
    )

    if np.mean(differences) < 0:
        print("Modified Z-Score has a lower average FPR.")

    elif np.mean(differences) > 0:
        print("Isolation Forest has a lower average FPR.")

    else:
        print("Both detectors have equal average FPR.")


# 6. Holm-Bonferroni correction

print("\n--- HOLM-BONFERRONI CORRECTION ---")

sorted_p_values = sorted(
    enumerate(p_values),
    key=lambda item: item[1]
)

adjusted_p_values = [0.0] * len(p_values)

previous_adjusted = 0.0
total_tests = len(p_values)

for rank, (original_index, p_value) in enumerate(sorted_p_values):

    adjusted = min(
        1.0,
        (total_tests - rank) * p_value
    )

    adjusted = max(adjusted, previous_adjusted)

    adjusted_p_values[original_index] = adjusted

    previous_adjusted = adjusted


for phase, adjusted_p in zip(phases, adjusted_p_values):

    print(f"{phase}: {adjusted_p:.10f}")

    if adjusted_p < 0.05:
        print("Result: Statistically significant after Holm correction.")
    else:
        print("Result: Not statistically significant after Holm correction.")