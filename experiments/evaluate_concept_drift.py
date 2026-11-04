import csv
import numpy as np

from statistics import median
from sklearn.ensemble import IsolationForest


# 1. Experiment configuration

TRIALS = 30

TRAINING_COUNT = 500
TEST_COUNT = 100

STATISTICAL_THRESHOLD = 3.5
ML_CONTAMINATION = 0.1

PHASES = {
    "Baseline": 40,
    "Mild Drift": 41,
    "Moderate Drift": 42,
    "Strong Drift": 43,
    "Severe Drift": 44
}

random_generator = np.random.default_rng(42)

experiment_records = []


# 2. Run repeated experiments

for trial in range(TRIALS):

    # Generate baseline training telemetry

    training_data = random_generator.normal(
        loc=40,
        scale=0.8,
        size=TRAINING_COUNT
    )

    # Calibrate statistical detector using baseline data

    baseline_median = median(training_data)

    baseline_mad = median(
        [abs(value - baseline_median) for value in training_data]
    )

    # Calibrate Isolation Forest using baseline data

    model = IsolationForest(
        contamination=ML_CONTAMINATION,
        random_state=trial
    )

    model.fit(training_data.reshape(-1, 1))


    # 3. Evaluate each operating phase

    for phase, normal_mean in PHASES.items():

        test_data = random_generator.normal(
            loc=normal_mean,
            scale=0.8,
            size=TEST_COUNT
        )

        ground_truth = [0] * TEST_COUNT


        # 4. Modified Z-Score using baseline statistics

        statistical_predictions = []

        for value in test_data:

            if baseline_mad == 0:
                score = (
                    0.0 if value == baseline_median
                    else float("inf")
                )
            else:
                score = (
                    0.6745
                    * (value - baseline_median)
                    / baseline_mad
                )

            statistical_predictions.append(
                int(abs(score) > STATISTICAL_THRESHOLD)
            )


        # 5. Isolation Forest using baseline model

        ml_predictions = [
            int(prediction == -1)
            for prediction in model.predict(
                test_data.reshape(-1, 1)
            )
        ]


        # 6. Calculate false alarm metrics

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
                false_positives / TEST_COUNT
            )

            experiment_records.append({
                "trial": trial + 1,
                "phase": phase,
                "normal_mean": normal_mean,
                "detector": name,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_positive_rate": false_positive_rate
            })


# 7. Summarize results

print("\nTAAHIS — CONCEPT DRIFT EXPERIMENT")

print(f"Trials: {TRIALS}")
print(f"Baseline training observations: {TRAINING_COUNT}")
print(f"Test observations per phase: {TEST_COUNT}")

for phase in PHASES:

    print(f"\n--- {phase.upper()} ---")

    for detector in [
        "Modified Z-Score",
        "Isolation Forest"
    ]:

        scores = [
            record["false_positive_rate"]
            for record in experiment_records
            if record["phase"] == phase
            and record["detector"] == detector
        ]

        print(
            f"{detector} Mean FPR: "
            f"{np.mean(scores):.4f}"
        )


# 8. Export results

with open(
    "data/concept_drift_results.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "trial",
        "phase",
        "normal_mean",
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
print("File: data/concept_drift_results.csv")