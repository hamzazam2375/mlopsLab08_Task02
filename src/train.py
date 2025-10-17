import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import yaml
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/house_data.csv")
    parser.add_argument("--model_out", type=str, default="models/house_model.pkl")
    parser.add_argument("--metrics_out", type=str, default="metrics.json")
    args = parser.parse_args()

    # Load parameters from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Load dataset
    df = pd.read_csv(args.data)

    # 🔹 Keep only numeric columns (drop URLs, text, etc.)
    df = df.select_dtypes(include=["number"])

    # Ensure dataset has enough columns
    if df.shape[1] < 2:
        raise ValueError("Dataset has insufficient numeric columns for training.")

    # Split features and target
    X = df.drop(columns=["price"], errors="ignore")
    y = df["price"] if "price" in df.columns else df.iloc[:, -1]

    # Save feature names for Flask app
    os.makedirs("models", exist_ok=True)
    import pickle
    pickle.dump(list(X.columns), open("models/model_features.pkl", "wb"))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"]
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=params["train"]["random_state"]
    )
    model.fit(X_train, y_train)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, args.model_out)

    # Evaluate
    preds = model.predict(X_test)
    metrics = {
        "r2_score": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds)
    }

    # Save metrics
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Training complete!")
    print("Model saved to:", args.model_out)
    print("Metrics:", metrics)
