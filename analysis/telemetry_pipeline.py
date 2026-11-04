from math import isfinite

from analysis.statistical_analyzer import detect_anomalies
from analysis.ml_analyzer import detect_ml_anomalies


def analyze_parameter(records, parameter_name):
    """
    Apply statistical and ML anomaly detection
    to one telemetry parameter.

    records:
        (timestamp, parameter, value, subsystem)
    """

    selected_records = [
        record
        for record in records
        if record[1] == parameter_name
    ]

    if len(selected_records) < 3:
        raise ValueError(
            f"At least 3 {parameter_name} records are required"
        )

    values = [record[2] for record in selected_records]

    # 1. Statistical detection handles invalid values.
    statistical_results = detect_anomalies(values)

    # 2. Identify valid numeric observations for ML.
    valid_indices = []
    valid_values = []

    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if isfinite(number):
            valid_indices.append(index)
            valid_values.append(number)

    # 3. Run ML only on valid observations.
    ml_lookup = {}

    if len(valid_values) >= 3:
        ml_results = detect_ml_anomalies(valid_values)

        for index, result in zip(valid_indices, ml_results):
            ml_lookup[index] = result

    # 4. Combine both detector results.
    report = []

    for index, (record, statistical) in enumerate(
        zip(selected_records, statistical_results)
    ):
        timestamp, parameter, value, subsystem = record

        ml = ml_lookup.get(index)

        ml_status = ml["status"] if ml else "InsufficientData"
        ml_score = ml["ml_score"] if ml else None
        ml_is_anomaly = ml["is_anomaly"] if ml else False

        statistical_is_anomaly = statistical["is_anomaly"]

        report.append({
            "timestamp": timestamp,
            "parameter": parameter,
            "value": statistical["value"],
            "subsystem": subsystem,

            "statistical_status": statistical["status"],
            "modified_z_score": statistical["modified_z_score"],
            "statistical_is_anomaly": statistical_is_anomaly,

            "ml_status": ml_status,
            "ml_score": ml_score,
            "ml_is_anomaly": ml_is_anomaly,

            # Combined result: either detector flags an anomaly.
            "is_anomaly": (
                statistical_is_anomaly or ml_is_anomaly
            ),
        })

    return report