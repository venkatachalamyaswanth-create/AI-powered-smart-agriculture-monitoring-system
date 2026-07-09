"""
generate_dataset.py
--------------------
Generates a synthetic soil + environmental sensor dataset for the
mini project: "AI-Based Soil & Sensor Data Driven Crop Yield and
Health Prediction".

Features (typical IoT / soil-sensor readings):
    N, P, K            - Soil Nitrogen, Phosphorus, Potassium (kg/ha)
    pH                 - Soil pH
    soil_moisture       - Soil moisture (%)
    temperature         - Ambient temperature (deg C)
    humidity            - Relative humidity (%)
    rainfall            - Rainfall (mm)

Targets:
    yield_kg_per_ha     - Continuous target for regression
    health_status        - Categorical target for classification
                            (Healthy / Stressed / Poor)

The relationships below are designed to be agronomically plausible
(e.g. yield rises with balanced NPK and moisture, falls with pH far
from neutral, extreme temperature, or drought/flood conditions) with
added Gaussian noise so the ML models have a realistic (not trivial)
learning problem.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_SAMPLES = 1200


def generate():
    N = np.random.normal(80, 20, N_SAMPLES).clip(0, 150)
    P = np.random.normal(45, 15, N_SAMPLES).clip(0, 100)
    K = np.random.normal(50, 18, N_SAMPLES).clip(0, 120)
    pH = np.random.normal(6.5, 0.8, N_SAMPLES).clip(4.0, 9.0)
    soil_moisture = np.random.normal(35, 12, N_SAMPLES).clip(5, 80)
    temperature = np.random.normal(27, 5, N_SAMPLES).clip(10, 45)
    humidity = np.random.normal(65, 15, N_SAMPLES).clip(20, 100)
    rainfall = np.random.exponential(80, N_SAMPLES).clip(0, 400)

    # --- Agronomic scoring function (drives yield) ---
    npk_score = 1 - (np.abs(N - 90) / 150 + np.abs(P - 50) / 100 + np.abs(K - 60) / 120) / 3
    ph_score = 1 - np.abs(pH - 6.5) / 2.5
    moisture_score = 1 - np.abs(soil_moisture - 40) / 40
    temp_score = 1 - np.abs(temperature - 25) / 20
    rain_score = 1 - np.abs(rainfall - 100) / 300

    composite = (0.30 * npk_score + 0.20 * ph_score + 0.25 * moisture_score +
                 0.15 * temp_score + 0.10 * rain_score).clip(0, 1)

    noise = np.random.normal(0, 0.08, N_SAMPLES)
    composite = (composite + noise).clip(0, 1)

    yield_kg_per_ha = (composite * 4500 + 500).round(1)  # 500 - 5000 kg/ha

    health_status = pd.cut(
        composite,
        bins=[-0.01, 0.62, 0.75, 1.01],
        labels=["Poor", "Stressed", "Healthy"]
    )

    df = pd.DataFrame({
        "N": N.round(1), "P": P.round(1), "K": K.round(1),
        "pH": pH.round(2), "soil_moisture": soil_moisture.round(1),
        "temperature": temperature.round(1), "humidity": humidity.round(1),
        "rainfall": rainfall.round(1),
        "yield_kg_per_ha": yield_kg_per_ha,
        "health_status": health_status.astype(str)
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/mini_project/data/sensor_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())
    print("\nClass balance:\n", df["health_status"].value_counts())
