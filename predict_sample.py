"""
predict_sample.py
------------------
Demonstrates using the trained models to make a prediction for a new
soil/sensor reading (e.g. as would arrive from a field IoT node).
"""

import joblib
import pandas as pd

MODEL_DIR = "/home/claude/mini_project/models"
FEATURES = ["N", "P", "K", "pH", "soil_moisture", "temperature", "humidity", "rainfall"]

yield_model = joblib.load(f"{MODEL_DIR}/yield_regressor.pkl")
yield_scaler = joblib.load(f"{MODEL_DIR}/yield_scaler.pkl")
health_model = joblib.load(f"{MODEL_DIR}/health_classifier.pkl")
health_scaler = joblib.load(f"{MODEL_DIR}/health_scaler.pkl")
label_encoder = joblib.load(f"{MODEL_DIR}/health_label_encoder.pkl")

# Example new sensor readings from two field nodes
new_readings = pd.DataFrame([
    {"N": 88, "P": 48, "K": 58, "pH": 6.4, "soil_moisture": 38,
     "temperature": 26, "humidity": 60, "rainfall": 95},   # well-balanced field
    {"N": 30, "P": 15, "K": 20, "pH": 4.8, "soil_moisture": 12,
     "temperature": 39, "humidity": 30, "rainfall": 5},    # stressed / drought field
], columns=FEATURES)

yield_pred = yield_model.predict(yield_scaler.transform(new_readings))
health_pred = label_encoder.inverse_transform(
    health_model.predict(health_scaler.transform(new_readings))
)

for i, row in new_readings.iterrows():
    print(f"\nSensor Node {i+1}: {row.to_dict()}")
    print(f"  -> Predicted Yield: {yield_pred[i]:.1f} kg/ha")
    print(f"  -> Predicted Health Status: {health_pred[i]}")
