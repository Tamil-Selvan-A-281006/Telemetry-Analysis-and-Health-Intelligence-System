from analysis.statistical_analyzer import detect_anomalies


def analyze_parameter(records, parameter_name):
    """
    Apply statistical anomaly detection to one telemetry parameter.

    records: iterable of (timestamp, parameter, value, subsystem)
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

    results = detect_anomalies(
        [record[2] for record in selected_records]
    )

    report = []

    for record, result in zip(selected_records, results):
        timestamp, parameter, value, subsystem = record

        report.append({
            "timestamp": timestamp,
            "parameter": parameter,
            "value": result["value"],
            "subsystem": subsystem,
            "statistical_status": result["status"],
            "modified_z_score": result["modified_z_score"],
            "is_anomaly": result["is_anomaly"],
        })

    return report