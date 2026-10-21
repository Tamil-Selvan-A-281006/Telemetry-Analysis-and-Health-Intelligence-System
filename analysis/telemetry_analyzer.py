import math

def validate_telemetry_record(timestamp, parameter, value, subsystem):
    if timestamp is None:
        return False

    if not isinstance(parameter, str) or not parameter.strip():
        return False

    if not isinstance(subsystem, str) or not subsystem.strip():
        return False

    if value is None:
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(numeric_value):
        return False

    return True

def analyze_temperature(value):
    if value>45:
        status="Critical"
    elif value>40:
        status="High"
    else:
        status="Normal"
    return status

def analyze_voltage(value):
    if value < 22:
        status = "Critical"
    elif value < 24:
        status = "Low"
    elif value <= 30:
        status = "Normal"
    elif value <= 32:
        status = "High"
    else:
        status = "Critical"
    return status

def analyze_pressure(value):
    if value < 90:
        status = "Critical"
    elif value < 95:
        status = "Low"
    elif value <= 105:
        status = "Normal"
    elif value < 110:
        status = "High"
    else:
        status = "Critical"
    return status

def analyze_attitude(target, actual, angular_velocity,subsystem,timestamp):
    orientation_error=abs(target-actual)
    if orientation_error>5:
        status='Critical'
    elif orientation_error>2:
        status='High'
    else:
        status='Normal'
    return {"target_orientation":target,"actual_orientation":actual,"timestamp":timestamp,"angular_velocity":angular_velocity,"subsystem":subsystem,"orientation_error":float(orientation_error),"status":status}

def calculate_rate_of_change(current_actual,previous_actual,current_timestamp,previous_timestamp):
    if previous_timestamp is None or previous_actual is None:
        return None
    time_difference = current_timestamp - previous_timestamp

    if time_difference <= 0:
        return None
    rate=(current_actual-float(previous_actual))/time_difference
    return rate

def analyze_telemetry(data):
    analysis_report = []

    for timestamp,parameter, value, subsystem in data:
        if not validate_telemetry_record(
        timestamp, parameter, value, subsystem
        ):
            continue

        if parameter == "temperature":
            status=analyze_temperature(value)
            
        elif parameter == "voltage":
            status=analyze_voltage(value)

        elif parameter == "pressure":
            status=analyze_pressure(value)
        else:
            continue
        analysis_report.append({
        "parameter": parameter,
        "value": float(value),
        "subsystem": subsystem,
        "status": status,
        "timestamp":timestamp
        })
    return analysis_report
