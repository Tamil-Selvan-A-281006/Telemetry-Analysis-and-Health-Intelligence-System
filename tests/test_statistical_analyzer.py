import pytest

from analysis.statistical_analyzer import detect_anomalies


def test_detects_extreme_outlier():
    results = detect_anomalies([10, 11, 10, 12, 11, 100])

    assert results[-1]["is_anomaly"] is True
    assert results[-1]["status"] == "Anomaly"


def test_typical_values_are_not_anomalies():
    results = detect_anomalies([10, 11, 10, 12, 11])

    assert all(not result["is_anomaly"] for result in results)


def test_handles_non_numeric_value():
    results = detect_anomalies([10, 11, "invalid", 12])

    assert results[2]["status"] == "Invalid"
    assert results[2]["modified_z_score"] is None


def test_rejects_too_few_valid_values():
    with pytest.raises(ValueError):
        detect_anomalies([10, 11])


def test_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        detect_anomalies([10, 11, 12], threshold=0)

def test_detects_negative_extreme_outlier():
    results = detect_anomalies(
        [10, 11, 10, 12, 11, -100]
    )

    assert results[-1]["is_anomaly"] is True


def test_detects_outlier_with_zero_mad():
    results = detect_anomalies(
        [10, 10, 10, 10, 100]
    )

    assert results[-1]["is_anomaly"] is True


def test_returns_expected_number_of_results():
    values = [10, 11, 10, 12, 11]

    results = detect_anomalies(values)

    assert len(results) == len(values)