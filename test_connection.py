from dotenv import load_dotenv
from analysis.telemetry_pipeline import analyze_parameter
from analysis.telemetry_analyzer import analyze_telemetry,analyze_attitude,calculate_rate_of_change
import os
import psycopg
load_dotenv()
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
name = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
connection = psycopg.connect(
    host=host,
    port=port,
    dbname=name,
    user=user,
    password=password
)
def check_temperature():
    sql_command="""
    SELECT value,subsystem FROM telemetry
    WHERE parameter='temperature';
    """
    cursor=connection.cursor()
    retrieved_data=cursor.execute(sql_command)
    for i in retrieved_data:
        if i[0]>45:
            print(f"🚨 CRITICAL TEMPERATURE \nSubsystem: {i[1]} \nTemperature: {i[0]}")
        elif i[0]>40:
            print(f"⚠️ HIGH TEMPERATURE \nSubsystem: {i[1]} \nTemperature: {i[0]}")
        else:
            print(f"✅ NORMAL TEMPERATURE\nSubsystem: {i[1]}\nTemperature: {i[0]}")
    connection.commit()
def get_attitude_data():
    cursor = connection.cursor()

    sql_command = """
SELECT target_orientation,
       actual_orientation,
       angular_velocity,
       subsystem,
       timestamp
FROM attitude_telemetry
ORDER BY timestamp, attitude_id;
    """

    cursor.execute(sql_command)
    retrieved_data = cursor.fetchall()

    return retrieved_data

def get_telemetry_data():
    cursor = connection.cursor()
    sql_command = """
    SELECT timestamp, parameter, value, subsystem
    FROM telemetry
    ORDER BY timestamp, telemetry_id;
    """
    cursor.execute(sql_command)
    retrieved_data=cursor.fetchall()
    return retrieved_data

def run_telemetry_analysis():
    analysis_report = []

    # 1. Retrieve database records
    attitude_data = get_attitude_data()
    telemetry_data = get_telemetry_data()

    # 2. Rule-based analysis
    telemetry_results = analyze_telemetry(telemetry_data)

    # 3. Statistical anomaly detection
    statistical_results = analyze_parameter(
        telemetry_data,
        "temperature"
    )

    # 4. Create a lookup for statistical results
    statistical_lookup = {
        (
            item["timestamp"],
            item["parameter"],
            item["subsystem"],
            item["value"]
        ): item
        for item in statistical_results
    }

    # 5. Merge rule-based and statistical results
    for item in telemetry_results:

        key = (
            item["timestamp"],
            item["parameter"],
            item["subsystem"],
            item["value"]
        )

        statistical_result = statistical_lookup.get(key)

        if statistical_result:
            item["statistical_status"] = (
                statistical_result["statistical_status"]
            )
            item["modified_z_score"] = (
                statistical_result["modified_z_score"]
            )
            item["is_anomaly"] = (
                statistical_result["is_anomaly"]
            )

    analysis_report.extend(telemetry_results)

    # 6. Attitude analysis
    previous_actual = None
    previous_timestamp = None

    for target, actual, angular_velocity, subsystem, timestamp in attitude_data:

        analysis_result = analyze_attitude(
            target,
            actual,
            angular_velocity,
            subsystem,
            timestamp
        )

        current_actual = float(actual)
        current_timestamp = timestamp.timestamp()

        orientation_rate = calculate_rate_of_change(
            current_actual,
            previous_actual,
            current_timestamp,
            previous_timestamp
        )

        analysis_result["orientation_rate"] = orientation_rate

        analysis_report.append(analysis_result)

        previous_actual = current_actual
        previous_timestamp = current_timestamp

    return analysis_report

def display_alert(result):
    if not result:
        print("ℹ️ NO TEMPERATURE DATA AVAILABLE")
    for i in result:
        temperature = i["temperature"]
        status = i["status"]
        if status=="Critical":
            print(f"☠ CRITICAL TEMPERATURE:\nTemperature:{temperature}")
        elif status=="High":
            print(f"⚠️ HIGH TEMPERATURE:\nTemperature:{temperature}")
        else:
            print(f"✅ NORMAL TEMPERATURE:\nTemperature:{temperature}")

if __name__ == "__main__":
    report = run_telemetry_analysis()

    for result in report:
        print(result)