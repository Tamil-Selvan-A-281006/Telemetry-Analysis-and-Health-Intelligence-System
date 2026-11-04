from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

from analysis.ml_analyzer import detect_ml_anomalies


# 1. Experimental dataset

normal_readings = [
    38.5, 39.2, 39.4, 39.5, 39.6,
    39.7, 39.8, 39.9, 40.0, 40.1,
    40.2, 40.3, 40.4, 40.5, 40.6,
    40.7, 40.8, 41.0, 41.2, 41.3,
    41.5, 41.6, 41.7, 41.8
]

anomalies = [42.5, 43.0, 44.0, 45.0, 50.0, 55.0]

telemetry_values = normal_readings + anomalies

ground_truth = (
    [0] * len(normal_readings)
    + [1] * len(anomalies)
)


# 2. Test different contamination settings

contamination_values = [
    "auto",
    0.05,
    0.10,
    0.20,
    0.30
]


print("\nTAAHIS — ISOLATION FOREST CONTAMINATION EXPERIMENT")

print("Total observations:", len(telemetry_values))
print("Normal observations:", len(normal_readings))
print("Injected anomalies:", len(anomalies))


# 3. Evaluate each configuration

for contamination in contamination_values:

    results = detect_ml_anomalies(
        telemetry_values,
        contamination=contamination
    )

    predictions = [
        int(result["is_anomaly"])
        for result in results
    ]

        # Identify incorrectly classified observations

    print("\nMisclassified observations:")

    for value, actual, predicted in zip(
        telemetry_values,
        ground_truth,
        predictions
    ):

        if actual != predicted:

            print(
                f"Value: {value} | "
                f"Actual: {'Anomaly' if actual else 'Normal'} | "
                f"Predicted: {'Anomaly' if predicted else 'Normal'}"
            )

    tn, fp, fn, tp = confusion_matrix(
        ground_truth,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\n--- Contamination: {contamination} ---")

    print("True Positives:", tp)
    print("False Positives:", fp)
    print("True Negatives:", tn)
    print("False Negatives:", fn)

    print(
        "Precision:",
        round(precision_score(
            ground_truth, predictions, zero_division=0
        ), 4)
    )

    print(
        "Recall:",
        round(recall_score(
            ground_truth, predictions, zero_division=0
        ), 4)
    )

    print(
        "F1-Score:",
        round(f1_score(
            ground_truth, predictions, zero_division=0
        ), 4)
    )

    print(
        "Accuracy:",
        round(accuracy_score(
            ground_truth, predictions
        ), 4)
    )