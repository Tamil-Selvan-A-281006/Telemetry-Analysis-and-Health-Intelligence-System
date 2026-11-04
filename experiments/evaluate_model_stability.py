from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from statistics import mean, pstdev

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


# 2. Experimental configurations

random_states = [0, 7, 21, 42, 100, 2026]

contamination = 0.20

f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []


# 3. Run repeated experiments

print("\nTAAHIS — ISOLATION FOREST STABILITY EXPERIMENT")

print("Total observations:", len(telemetry_values))
print("Contamination:", contamination)

for seed in random_states:

    results = detect_ml_anomalies(
        telemetry_values,
        contamination=contamination,
        random_state=seed
    )

    predictions = [
        int(result["is_anomaly"])
        for result in results
    ]

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

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    accuracy_scores.append(accuracy)

    print(f"\nRandom State: {seed}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")


# 4. Calculate average performance

print("\n--- AVERAGE PERFORMANCE ---")

print(f"Mean Precision: {mean(precision_scores):.4f}")
print(f"Mean Recall:    {mean(recall_scores):.4f}")
print(f"Mean F1-Score:  {mean(f1_scores):.4f}")
print(f"Mean Accuracy:  {mean(accuracy_scores):.4f}")


# 5. Calculate performance variability

print("\n--- PERFORMANCE VARIABILITY ---")

print(f"F1 Standard Deviation: {pstdev(f1_scores):.4f}")
print(f"Precision Standard Deviation: {pstdev(precision_scores):.4f}")
print(f"Recall Standard Deviation: {pstdev(recall_scores):.4f}")
print(f"Accuracy Standard Deviation: {pstdev(accuracy_scores):.4f}")