from math import isfinite

from sklearn.ensemble import IsolationForest


def detect_ml_anomalies(
    values,
    contamination="auto",
    random_state=42
):
    """
    Detect anomalies in telemetry observations using Isolation Forest.

    Returns anomaly predictions and decision scores.
    """

    if contamination != "auto":
        if not isinstance(contamination, (int, float)):
            raise ValueError("Invalid contamination value")

        if not 0 < contamination <= 0.5:
            raise ValueError(
                "Contamination must be between 0 and 0.5"
            )

    numeric_values = []

    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("All telemetry values must be numeric")

        if not isfinite(number):
            raise ValueError("Telemetry values must be finite")

        numeric_values.append(number)

    if len(numeric_values) < 3:
        raise ValueError(
            "At least 3 observations are required"
        )

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )

    features = [[value] for value in numeric_values]

    predictions = model.fit_predict(features)
    scores = model.decision_function(features)

    results = []

    for value, prediction, score in zip(
        numeric_values,
        predictions,
        scores
    ):
        is_anomaly = prediction == -1

        results.append({
            "value": value,
            "ml_score": float(score),
            "is_anomaly": is_anomaly,
            "status": "Anomaly" if is_anomaly else "Normal"
        })

    return results