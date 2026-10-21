from analysis.telemetry_analyzer import (
    validate_telemetry_record,
    analyze_temperature,
    analyze_voltage,
    analyze_pressure,
    analyze_attitude,
    calculate_rate_of_change,
)

def test_temperature_normal():
    assert analyze_temperature(38.5) == "Normal"


def test_temperature_high():
    assert analyze_temperature(42.0) == "High"


def test_temperature_critical():
    assert analyze_temperature(46.0) == "Critical"


def test_voltage_normal():
    assert analyze_voltage(27.4) == "Normal"


def test_pressure_normal():
    assert analyze_pressure(101.2) == "Normal"


def test_attitude_normal():
    result = analyze_attitude(45, 45.5, 0.2, "attitude", None)

    assert result["orientation_error"] == 0.5
    assert result["status"] == "Normal"

def test_temperature_at_40_is_normal():
    assert analyze_temperature(40) == "Normal"


def test_temperature_at_45_is_high():
    assert analyze_temperature(45) == "High"


def test_temperature_above_45_is_critical():
    assert analyze_temperature(45.1) == "Critical"


def test_voltage_at_22_is_low():
    assert analyze_voltage(22) == "Low"


def test_voltage_at_24_is_normal():
    assert analyze_voltage(24) == "Normal"


def test_voltage_at_30_is_normal():
    assert analyze_voltage(30) == "Normal"


def test_voltage_at_32_is_high():
    assert analyze_voltage(32) == "High"


def test_pressure_at_95_is_normal():
    assert analyze_pressure(95) == "Normal"


def test_pressure_at_105_is_normal():
    assert analyze_pressure(105) == "Normal"


def test_pressure_at_110_is_critical():
    assert analyze_pressure(110) == "Critical"

def test_rate_of_change_first_reading():
    result = calculate_rate_of_change(
        current_actual=45.5,
        previous_actual=None,
        current_timestamp=100.0,
        previous_timestamp=None,
    )

    assert result is None


def test_rate_of_change_one_second():
    result = calculate_rate_of_change(
        current_actual=47.0,
        previous_actual=45.5,
        current_timestamp=101.0,
        previous_timestamp=100.0,
    )

    assert result == 1.5


def test_rate_of_change_zero_time_difference():
    result = calculate_rate_of_change(
        current_actual=47.0,
        previous_actual=45.5,
        current_timestamp=100.0,
        previous_timestamp=100.0,
    )

    assert result is None

def test_valid_telemetry_record():
    assert validate_telemetry_record(
        "2026-10-21T10:00:00",
        "temperature",
        42.0,
        "thermal",
    ) is True


def test_missing_value():
    assert validate_telemetry_record(
        "2026-10-21T10:00:00",
        "temperature",
        None,
        "thermal",
    ) is False


def test_non_numeric_value():
    assert validate_telemetry_record(
        "2026-10-21T10:00:00",
        "temperature",
        "unknown",
        "thermal",
    ) is False


def test_nan_value():
    assert validate_telemetry_record(
        "2026-10-21T10:00:00",
        "temperature",
        float("nan"),
        "thermal",
    ) is False


def test_missing_timestamp():
    assert validate_telemetry_record(
        None,
        "temperature",
        42.0,
        "thermal",
    ) is False
