from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found. Please train the model first.")

model = joblib.load(MODEL_PATH)

# ---------------- HOME / UI ----------------

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- REST API ----------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        required = [
            "age",
            "time_in_hospital",
            "num_lab_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient"
        ]

        # Validate input
        for r in required:
            if r not in data:
                return jsonify({"error": f"{r} missing"}), 400

        # Prepare feature array
        features = np.array([[data[k] for k in required]])

        # Predict probability
        probability = model.predict_proba(features)[0][1]
        label = 1 if probability > 0.5 else 0

        # Basic monitoring (log predictions)
        with open("predictions.log", "a") as f:
            f.write(str(probability) + "\n")

        return jsonify({
            "readmission_risk": float(probability),
            "prediction": int(label)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- UI Prediction ----------------

@app.route("/ui_predict", methods=["POST"])
def ui_predict():
    try:
        d = request.form

        features = np.array([[
            float(d["age"]),
            float(d["time_in_hospital"]),
            float(d["num_lab_procedures"]),
            float(d["num_medications"]),
            float(d["number_outpatient"]),
            float(d["number_emergency"]),
            float(d["number_inpatient"])
        ]])

        prob = model.predict_proba(features)[0][1]
        risk = round(prob * 100, 2)

        return render_template("index.html", risk=risk)

    except Exception as e:
        return render_template("index.html", risk="Error")

# ---------------- MAIN ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)