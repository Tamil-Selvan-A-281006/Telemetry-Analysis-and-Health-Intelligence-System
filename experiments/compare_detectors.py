from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from analysis.statistical_analyzer import detect_anomalies
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


# 2. Statistical detector

statistical_results = detect_anomalies(
    telemetry_values,
    threshold=2.5
)

statistical_predictions = [
    int(result["is_anomaly"])
    for result in statistical_results
]


# 3. Machine learning detector

ml_results = detect_ml_anomalies(telemetry_values)

ml_predictions = [
    int(result["is_anomaly"])
    for result in ml_results
]


# 4. Evaluation function

def evaluate_detector(name, actual, predicted):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    print(f"\n--- {name} ---")

    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")

    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(actual, predicted, zero_division=0):.4f}")
    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")


# 5. Compare both detectors

print("TAAHIS — ANOMALY DETECTOR COMPARISON")

print(f"Total observations: {len(telemetry_values)}")
print(f"Normal observations: {len(normal_readings)}")
print(f"Injected anomalies: {len(anomalies)}")

evaluate_detector(
    "MODIFIED Z-SCORE",
    ground_truth,
    statistical_predictions
)

evaluate_detector(
    "ISOLATION FOREST",
    ground_truth,
    ml_predictions
)


# 6. Prediction agreement

agreement = sum(
    a == b
    for a, b in zip(
        statistical_predictions,
        ml_predictions
    )
)

print("\n--- DETECTOR AGREEMENT ---")
print(f"Agreement: {agreement}/{len(ground_truth)}")
print(f"Agreement percentage: {agreement / len(ground_truth) * 100:.2f}%")