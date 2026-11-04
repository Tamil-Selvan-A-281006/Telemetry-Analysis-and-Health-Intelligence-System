from analysis.ml_analyzer import detect_ml_anomalies


temperature_readings = [
    39.2, 40.1, 39.8, 41.0, 40.5,
    39.7, 40.3, 41.2, 39.5, 40.8,
    39.9, 40.4, 41.1, 39.6, 40.2,
    40.7, 39.4, 41.3, 40.0, 40.6,
    55.0, 60.0
]

results = detect_ml_anomalies(
    temperature_readings,
    contamination="auto"
)

print("\n--- Isolation Forest Results ---")

for result in results:
    print(
        f"Value: {result['value']:5.1f} | "
        f"Score: {result['ml_score']:8.4f} | "
        f"Status: {result['status']}"
    )