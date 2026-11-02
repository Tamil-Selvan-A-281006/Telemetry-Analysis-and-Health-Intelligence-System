from analysis.telemetry_pipeline import analyze_parameter


def test_analyze_parameter_preserves_record_information():
    records = [
        ("t1", "temperature", 30, "thermal"),
        ("t2", "temperature", 31, "thermal"),
        ("t3", "temperature", 30, "thermal"),
        ("t4", "temperature", 32, "thermal"),
        ("t5", "temperature", 31, "thermal"),
        ("t6", "temperature", 100, "thermal"),
        ("t7", "voltage", 27, "power"),
    ]

    report = analyze_parameter(records, "temperature")

    assert len(report) == 6
    assert report[-1]["value"] == 100.0
    assert report[-1]["is_anomaly"] is True
    assert report[-1]["timestamp"] == "t6"
    assert all(
        item["parameter"] == "temperature"
        for item in report
    )

def test_pipeline_handles_invalid_value():

    records = [
        ("2026-10-21T10:00:00", "temperature", 10, "thermal"),
        ("2026-10-21T10:00:01", "temperature", 11, "thermal"),
        ("2026-10-21T10:00:02", "temperature", "invalid", "thermal"),
        ("2026-10-21T10:00:03", "temperature", 12, "thermal"),
    ]

    report = analyze_parameter(records, "temperature")

    assert report[2]["statistical_status"] == "Invalid"
    assert report[2]["value"] is None
    assert report[2]["is_anomaly"] is False
    