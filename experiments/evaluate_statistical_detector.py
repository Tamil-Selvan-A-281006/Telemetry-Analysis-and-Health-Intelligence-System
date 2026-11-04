from analysis.statistical_analyzer import detect_anomalies


# Normal simulated temperature readings
normal_readings = [
    38.5, 39.2, 39.4, 39.5, 39.6,
    39.7, 39.8, 39.9, 40.0, 40.1,
    40.2, 40.3, 40.4, 40.5, 40.6,
    40.7, 40.8, 41.0, 41.2, 41.3,
    41.5, 41.6, 41.7, 41.8
]

# Injected anomalies with increasing magnitudes
anomalies = [
    42.5,
    43.0,
    44.0,
    45.0,
    50.0,
    55.0
]
# Combine normal and abnormal readings
telemetry_values = normal_readings + anomalies

# Ground truth: 0 = Normal, 1 = Anomaly
ground_truth = (
    [0] * len(normal_readings)
    + [1] * len(anomalies)
)
assert len(telemetry_values) == len(ground_truth)

print("Normal readings:", len(normal_readings))
print("Injected anomalies:", len(anomalies))
print("Total readings:", len(telemetry_values))


print("Total readings:", len(telemetry_values))
print("Ground truth labels:", len(ground_truth))

# Run the statistical anomaly detector
predictions = detect_anomalies(telemetry_values, threshold=2.5)


# Display actual values and detector predictions
print("\n--- Anomaly Detection Results ---")

for value, actual, prediction in zip(
    telemetry_values,
    ground_truth,
    predictions
):
    print(
        f"Value: {value:5.1f} | "
        f"Ground Truth: {actual} | "
        f"Modified Z-score: {prediction['modified_z_score']:.4f} | "
        f"Prediction: {prediction['is_anomaly']}"
    )

# Convert detector results into binary predictions
predicted_labels = [
    1 if result["is_anomaly"] else 0
    for result in predictions
]


# Calculate confusion matrix values
TP = sum(
    actual == 1 and predicted == 1
    for actual, predicted in zip(ground_truth, predicted_labels)
)

FP = sum(
    actual == 0 and predicted == 1
    for actual, predicted in zip(ground_truth, predicted_labels)
)

TN = sum(
    actual == 0 and predicted == 0
    for actual, predicted in zip(ground_truth, predicted_labels)
)

FN = sum(
    actual == 1 and predicted == 0
    for actual, predicted in zip(ground_truth, predicted_labels)
)


print("\n--- Confusion Matrix Results ---")
print("True Positives:", TP)
print("False Positives:", FP)
print("True Negatives:", TN)
print("False Negatives:", FN)

# Calculate evaluation metrics

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

accuracy = (TP + TN) / len(ground_truth)


# Display evaluation results

print("\n--- Evaluation Metrics ---")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

# Threshold sensitivity experiment

print("\n--- Threshold Comparison ---")

thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]

for threshold in thresholds:

    results = detect_anomalies(
        telemetry_values,
        threshold=threshold
    )

    predicted_labels = [
        1 if result["is_anomaly"] else 0
        for result in results
    ]

    TP = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip(ground_truth, predicted_labels)
    )

    FP = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip(ground_truth, predicted_labels)
    )

    TN = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip(ground_truth, predicted_labels)
    )

    FN = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip(ground_truth, predicted_labels)
    )

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    accuracy = (TP + TN) / len(ground_truth)

    print(
        f"Threshold: {threshold:.1f} | "
        f"TP: {TP} | FP: {FP} | "
        f"FN: {FN} | TN: {TN} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1_score:.2f} | "
        f"Accuracy: {accuracy:.2f}"
    )