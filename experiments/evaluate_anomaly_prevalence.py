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

ANOMALY_COUNTS = [2, 5, 10, 20, 30]

STATISTICAL_THRESHOLD = 3.5
ML_CONTAMINATION = 0.1

random_generator = np.random.default_rng(42)

experiment_records = []


# 2. Test different anomaly proportions

for anomaly_count in ANOMALY_COUNTS:

    statistical_scores = []
    ml_scores = []

    for trial in range(TRIALS):

        # Generate normal telemetry

        normal_readings = random_generator.normal(
            loc=40,
            scale=0.8,
            size=NORMAL_COUNT
        )

        # Generate anomalous telemetry

        anomaly_readings = random_generator.normal(
            loc=46,
            scale=0.8,
            size=anomaly_count
        )

        telemetry_values = np.concatenate([
            normal_readings,
            anomaly_readings
        ])

        ground_truth = (
            [0] * NORMAL_COUNT
            + [1] * anomaly_count
        )

        # 3. Modified Z-Score

        statistical_results = detect_anomalies(
            telemetry_values,
            threshold=STATISTICAL_THRESHOLD
        )

        statistical_predictions = [
            int(result["is_anomaly"])
            for result in statistical_results
        ]

        # 4. Isolation Forest

        ml_results = detect_ml_anomalies(
            telemetry_values,
            contamination=ML_CONTAMINATION,
            random_state=trial
        )

        ml_predictions = [
            int(result["is_anomaly"])
            for result in ml_results
        ]

        # 5. Evaluate both detectors

        for detector, predictions in [
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
                "anomaly_count": anomaly_count,
                "anomaly_prevalence": (
                    anomaly_count /
                    (NORMAL_COUNT + anomaly_count)
                ),
                "trial": trial + 1,
                "detector": detector,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy
            })

            if detector == "Modified Z-Score":
                statistical_scores.append(f1)
            else:
                ml_scores.append(f1)

    # 6. Display results

    prevalence = (
        anomaly_count /
        (NORMAL_COUNT + anomaly_count)
    ) * 100

    print(f"\n--- ANOMALY COUNT: {anomaly_count} ---")
    print(f"Anomaly prevalence: {prevalence:.2f}%")

    print(
        f"Modified Z-Score Mean F1: "
        f"{np.mean(statistical_scores):.4f}"
    )

    print(
        f"Modified Z-Score F1 Std: "
        f"{np.std(statistical_scores):.4f}"
    )

    print(
        f"Isolation Forest Mean F1: "
        f"{np.mean(ml_scores):.4f}"
    )

    print(
        f"Isolation Forest F1 Std: "
        f"{np.std(ml_scores):.4f}"
    )


# 7. Export results

with open(
    "data/anomaly_prevalence_results.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "anomaly_count",
        "anomaly_prevalence",
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
print("File: data/anomaly_prevalence_results.csv")