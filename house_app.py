from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
import subprocess
import joblib

app = Flask(__name__)

# Base directory for this file so relative paths work regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model and features (absolute)
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")

# Sync latest model from DVC remote if needed
def fetch_latest_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print("Fetching model from DVC remote...")
        try:
            # run dvc pull but do not raise hard error if it fails (e.g., not a
            # dvc repo or DVC not installed). Just warn the user.
            res = subprocess.run(["dvc", "pull"], check=False, capture_output=True, text=True)
            if res.returncode != 0:
                print("Warning: 'dvc pull' failed or DVC not available:\n", res.stderr)
        except FileNotFoundError:
            print("Warning: 'dvc' executable not found. Skipping DVC fetch.")

# Load model and features
def load_model():
    fetch_latest_model()
    # The training script saves the trained estimator with joblib.dump and
    # the feature names with pickle.dump. Use joblib.load for the model and
    # pickle for features to avoid ambiguities.
    model = None
    features = None

    # Try loading the model with joblib first (correct for sklearn models)
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        # fallback to pickle if joblib can't read it
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

    # Validate that loaded model has predict
    if not (hasattr(model, "predict") and callable(getattr(model, "predict"))):
        raise RuntimeError(
            f"The object loaded from {MODEL_PATH} does not look like a trained model (missing 'predict'). Got type: {type(model)}"
        )

    # Load feature names (expected to be a list saved with pickle)
    try:
        with open(FEATURES_PATH, "rb") as f:
            features_obj = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load features from {FEATURES_PATH}: {e}")

    # Normalize features to a list of strings
    if isinstance(features_obj, (list, tuple)):
        features = list(features_obj)
    else:
        try:
            import numpy as _np
            if isinstance(features_obj, _np.ndarray):
                features = [str(x) for x in features_obj.tolist()]
            else:
                raise TypeError()
        except Exception:
            raise RuntimeError(
                f"The features object loaded from {FEATURES_PATH} doesn't look like a list of feature names (type={type(features_obj)})."
            )

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
