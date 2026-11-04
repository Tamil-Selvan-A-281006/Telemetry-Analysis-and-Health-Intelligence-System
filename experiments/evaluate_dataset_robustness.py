import csv
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from analysis.statistical_analyzer import detect_anomalies
from analysis.ml_analyzer import detect_ml_anomalies


# 1. Experiment configuration

TRIALS = 30
NORMAL_COUNT = 100
ANOMALY_COUNT = 10

STATISTICAL_THRESHOLD = 3.5
ML_CONTAMINATION = 0.1

random_generator = np.random.default_rng(42)


# 2. Store results

statistical_scores = []
ml_scores = []

experiment_records = []


# 3. Run multiple experiments

for trial in range(TRIALS):

    # Generate normal telemetry readings
    normal_readings = random_generator.normal(
        loc=40,
        scale=0.8,
        size=NORMAL_COUNT
    )

    # Generate injected anomalous readings
    anomaly_readings = random_generator.normal(
        loc=45,
        scale=1.0,
        size=ANOMALY_COUNT
    )

    telemetry_values = np.concatenate([
        normal_readings,
        anomaly_readings
    ])

    ground_truth = (
        [0] * NORMAL_COUNT
        + [1] * ANOMALY_COUNT
    )

    # 4. Statistical detection

    statistical_results = detect_anomalies(
        telemetry_values,
        threshold=STATISTICAL_THRESHOLD
    )

    statistical_predictions = [
        int(result["is_anomaly"])
        for result in statistical_results
    ]

    # 5. Machine learning detection

    ml_results = detect_ml_anomalies(
        telemetry_values,
        contamination=ML_CONTAMINATION,
        random_state=trial
    )

    ml_predictions = [
        int(result["is_anomaly"])
        for result in ml_results
    ]

    # 6. Evaluate both detectors

    for name, predictions in [
        ("Modified Z-Score", statistical_predictions),
        ("Isolation Forest", ml_predictions)
    ]:

        precision = precision_score(
            ground_truth,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            ground_truth,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            ground_truth,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            ground_truth,
            predictions
        )

        experiment_records.append({
            "trial": trial + 1,
            "detector": name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        })

        if name == "Modified Z-Score":
            statistical_scores.append(f1)
        else:
            ml_scores.append(f1)

# 7. Summarize results

print("\nTAAHIS — DATASET ROBUSTNESS EXPERIMENT")

print(f"Trials: {TRIALS}")
print(f"Normal observations per trial: {NORMAL_COUNT}")
print(f"Injected anomalies per trial: {ANOMALY_COUNT}")


print("\n--- MODIFIED Z-SCORE ---")

print(f"Mean F1-score: {np.mean(statistical_scores):.4f}")
print(f"F1 standard deviation: {np.std(statistical_scores):.4f}")


print("\n--- ISOLATION FOREST ---")

print(f"Mean F1-score: {np.mean(ml_scores):.4f}")
print(f"F1 standard deviation: {np.std(ml_scores):.4f}")

# 8. Export experimental results

with open(
    "data/robustness_experiment_results.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "trial",
        "detector",
        "precision",
        "recall",
        "f1_score",
        "accuracy"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(experiment_records)

print("\nResults exported successfully.")
print("File: data/robustness_experiment_results.csv")