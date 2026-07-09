"""
train_model.py
---------------
Mini Project: AI-Based Soil & Sensor Data Driven Crop Yield & Health Prediction

Pipeline:
  1. Load sensor dataset
  2. Preprocess (scale features)
  3. Train/test split
  4. Train RandomForestRegressor -> predict yield_kg_per_ha
  5. Train RandomForestClassifier -> predict health_status (Healthy/Stressed/Poor)
  6. Evaluate both models (RMSE, R2, Accuracy, F1, Confusion Matrix)
  7. Save plots (correlation heatmap, feature importance, actual vs predicted,
     confusion matrix) and trained models to disk
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, f1_score, classification_report,
                              confusion_matrix)

DATA_PATH = "/home/claude/mini_project/data/sensor_dataset.csv"
OUT_DIR = "/home/claude/mini_project/outputs"
MODEL_DIR = "/home/claude/mini_project/models"
FEATURES = ["N", "P", "K", "pH", "soil_moisture", "temperature", "humidity", "rainfall"]

sns.set_theme(style="whitegrid")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def correlation_heatmap(df):
    plt.figure(figsize=(8, 6))
    corr = df[FEATURES + ["yield_kg_per_ha"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()


def run_regression(df, scaler):
    X = df[FEATURES]
    y = df["yield_kg_per_ha"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[Regression - Yield Prediction]\n  RMSE: {rmse:.2f} kg/ha\n  MAE: {mae:.2f} kg/ha\n  R2: {r2:.3f}")

    # Actual vs Predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5, color="seagreen")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Ideal fit")
    plt.xlabel("Actual Yield (kg/ha)")
    plt.ylabel("Predicted Yield (kg/ha)")
    plt.title(f"Actual vs Predicted Yield (R2 = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yield_actual_vs_predicted.png", dpi=150)
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="darkgoldenrod")
    plt.title("Feature Importance - Yield Prediction (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yield_feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(model, f"{MODEL_DIR}/yield_regressor.pkl")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_classification(df, scaler):
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["health_status"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"\n[Classification - Crop Health Status]\n  Accuracy: {acc:.3f}\n  Weighted F1: {f1:.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Crop Health Status (Acc = {acc:.3f})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/health_confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(model, f"{MODEL_DIR}/health_classifier.pkl")
    joblib.dump(le, f"{MODEL_DIR}/health_label_encoder.pkl")
    return {"accuracy": acc, "f1": f1}


def main():
    df = load_data()
    correlation_heatmap(df)

    scaler_reg = StandardScaler()
    reg_metrics = run_regression(df, scaler_reg)
    joblib.dump(scaler_reg, f"{MODEL_DIR}/yield_scaler.pkl")

    scaler_clf = StandardScaler()
    clf_metrics = run_classification(df, scaler_clf)
    joblib.dump(scaler_clf, f"{MODEL_DIR}/health_scaler.pkl")

    summary = {**{"yield_" + k: v for k, v in reg_metrics.items()},
               **{"health_" + k: v for k, v in clf_metrics.items()}}
    pd.Series(summary).to_csv(f"{OUT_DIR}/metrics_summary.csv")
    print("\nAll models, metrics, and plots saved successfully.")


if __name__ == "__main__":
    main()
