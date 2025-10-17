from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
import subprocess

app = Flask(__name__)

# Paths to model and features
MODEL_PATH = "models/house_model.pkl"
FEATURES_PATH = "models/model_features.pkl"

# Sync latest model from DVC remote if needed
def fetch_latest_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print("Fetching model from DVC remote...")
        subprocess.run(["dvc", "pull"], check=True)

# Load model and features
def load_model():
    fetch_latest_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    return model, features

@app.route("/", methods=["GET", "POST"])
def predict():
    model, features = load_model()
    prediction = None
    if request.method == "POST":
        input_data = [float(request.form.get(f, 0)) for f in features]
        df = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(df)[0]
    return render_template("index.html", prediction=prediction, features=features)

if __name__ == "__main__":
    app.run(debug=True)
