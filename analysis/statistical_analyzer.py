from statistics import median
from math import isfinite


def detect_anomalies(values, threshold=3.5):
    """
    Detect outliers using the modified Z-score based on MAD.

    This initial version evaluates a batch of numeric observations.
    It is not yet a rolling or online detector.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    numeric_values = []

    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            numeric_values.append(None)
            continue

        numeric_values.append(number if isfinite(number) else None)

    valid_values = [value for value in numeric_values if value is not None]

    if len(valid_values) < 3:
        raise ValueError("At least 3 valid numeric observations are required")

    center = median(valid_values)
    absolute_deviations = [
        abs(value - center) for value in valid_values
    ]
    mad = median(absolute_deviations)

    results = []

    for value in numeric_values:
        if value is None:
            results.append({
                "value": value,
                "modified_z_score": None,
                "is_anomaly": False,
                "status": "Invalid"
            })
            continue

        if mad == 0:
            score = 0.0 if value == center else float("inf")
        else:
            score = 0.6745 * (value - center) / mad

        is_anomaly = abs(score) > threshold

        results.append({
            "value": value,
            "modified_z_score": score,
            "is_anomaly": is_anomaly,
            "status": "Anomaly" if is_anomaly else "Normal"
        })

    return results