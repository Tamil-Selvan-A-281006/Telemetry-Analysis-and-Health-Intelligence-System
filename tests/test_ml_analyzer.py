import pytest

from analysis.ml_analyzer import detect_ml_anomalies


def test_returns_correct_number_of_results():
    values = [39.2, 40.1, 39.8, 41.0, 55.0]

    results = detect_ml_anomalies(values)

    assert len(results) == len(values)


def test_result_contains_required_fields():
    values = [39.2, 40.1, 39.8, 41.0]

    results = detect_ml_anomalies(values)

    for result in results:
        assert "value" in result
        assert "ml_score" in result
        assert "is_anomaly" in result
        assert "status" in result


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        detect_ml_anomalies([39.2, "invalid", 40.1])


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError):
        detect_ml_anomalies([39.2, float("nan"), 40.1])


def test_insufficient_observations_are_rejected():
    with pytest.raises(ValueError):
        detect_ml_anomalies([39.2, 40.1])


def test_invalid_contamination_is_rejected():
    with pytest.raises(ValueError):
        detect_ml_anomalies(
            [39.2, 40.1, 39.8],
            contamination=0.8
        )


def test_reproducible_predictions():
    values = [39.2, 40.1, 39.8, 41.0, 55.0]

    first = detect_ml_anomalies(values)
    second = detect_ml_anomalies(values)

    assert [
        result["is_anomaly"] for result in first
    ] == [
        result["is_anomaly"] for result in second
    ]