import csv
import numpy as np

from analysis.statistical_analyzer import detect_anomalies
from analysis.ml_analyzer import detect_ml_anomalies


# 1. Experiment configuration

TRIALS = 30
NORMAL_COUNT = 100

STATISTICAL_THRESHOLD = 3.5
ML_CONTAMINATION = 0.1

random_generator = np.random.default_rng(42)

experiment_records = []

statistical_fpr_scores = []
ml_fpr_scores = []


# 2. Run experiments

for trial in range(TRIALS):

    # Generate only normal telemetry readings
    normal_readings = random_generator.normal(
        loc=40,
        scale=0.8,
        size=NORMAL_COUNT
    )

    # Every observation is genuinely normal
    ground_truth = [0] * NORMAL_COUNT

    # 3. Modified Z-Score

    statistical_results = detect_anomalies(
        normal_readings,
        threshold=STATISTICAL_THRESHOLD
    )

    statistical_predictions = [
        int(result["is_anomaly"])
        for result in statistical_results
    ]

    # 4. Isolation Forest

    ml_results = detect_ml_anomalies(
        normal_readings,
        contamination=ML_CONTAMINATION,
        random_state=trial
    )

    ml_predictions = [
        int(result["is_anomaly"])
        for result in ml_results
    ]

    # 5. Evaluate false alarm rates

    for name, predictions in [
        ("Modified Z-Score", statistical_predictions),
        ("Isolation Forest", ml_predictions)
    ]:

        false_positives = sum(
            actual == 0 and predicted == 1
            for actual, predicted in zip(
                ground_truth,
                predictions
            )
        )

        true_negatives = sum(
            actual == 0 and predicted == 0
            for actual, predicted in zip(
                ground_truth,
                predictions
            )
        )

        false_positive_rate = (
            false_positives / (false_positives + true_negatives)
        )

        experiment_records.append({
            "trial": trial + 1,
            "detector": name,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_positive_rate": false_positive_rate
        })

        if name == "Modified Z-Score":
            statistical_fpr_scores.append(false_positive_rate)
        else:
            ml_fpr_scores.append(false_positive_rate)


# 6. Summarize results

print("\nTAAHIS — FALSE ALARM RATE EXPERIMENT")

print(f"Trials: {TRIALS}")
print(f"Normal observations per trial: {NORMAL_COUNT}")
print("Injected anomalies: 0")


print("\n--- MODIFIED Z-SCORE ---")

print(f"Mean FPR: {np.mean(statistical_fpr_scores):.4f}")
print(f"FPR Standard Deviation: {np.std(statistical_fpr_scores):.4f}")


print("\n--- ISOLATION FOREST ---")

print(f"Mean FPR: {np.mean(ml_fpr_scores):.4f}")
print(f"FPR Standard Deviation: {np.std(ml_fpr_scores):.4f}")


# 7. Export results

with open(
    "data/false_alarm_results.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "trial",
        "detector",
        "false_positives",
        "true_negatives",
        "false_positive_rate"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(experiment_records)


print("\nResults exported successfully.")
print("File: data/false_alarm_results.csv")