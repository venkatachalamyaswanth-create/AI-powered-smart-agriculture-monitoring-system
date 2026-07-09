# AI-Based Soil & Sensor Data Driven Crop Yield & Health Prediction
### (Mini Project — Precision Agriculture / AIML)

## Overview
This mini project predicts **crop yield (kg/hectare)** and **crop health status**
(Healthy / Stressed / Poor) from soil and environmental IoT sensor readings
(N, P, K, pH, soil moisture, temperature, humidity, rainfall) using Random Forest
machine learning models.

## Folder Structure
```
mini_project/
├── data/
│   └── sensor_dataset.csv        # Generated synthetic sensor dataset (1200 samples)
├── models/                       # Saved trained models + scalers (.pkl)
├── outputs/                      # Evaluation plots + metrics summary
├── src/
│   ├── generate_dataset.py       # Creates the synthetic sensor dataset
│   ├── train_model.py            # Preprocessing, training, evaluation, plots
│   └── predict_sample.py         # Demo: predict yield & health for new readings
└── README.md
```

## How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib

python src/generate_dataset.py    # Step 1: create dataset
python src/train_model.py         # Step 2: train + evaluate models
python src/predict_sample.py      # Step 3: run inference on sample sensor readings
```

## Models
- **Yield Prediction:** RandomForestRegressor
- **Health Classification:** RandomForestClassifier (Healthy / Stressed / Poor)

## Notes
The dataset is synthetically generated with agronomically-informed relationships
(e.g., yield rises with balanced NPK, pH near neutral, moderate moisture/temperature)
plus realistic noise, since live multispectral/IoT field data was not available for
this mini-project scope. The same pipeline works unmodified on real sensor data —
only `generate_dataset.py` would be swapped for a real data loader.
