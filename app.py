import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# -------------------------------
# Load ML Model Once at Startup
# -------------------------------
try:
    with open("vehicle_health_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Loading Failed:", e)
    model = None


# -------------------------------
# Health Prediction Endpoint
# -------------------------------
@app.route("/api/telemetry", methods=["POST"])
def receive_telemetry():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    obd = data.get("obd", {})

    try:
        # IMPORTANT:
        # Feature order must match your model training
        rpm = float(obd.get("rpm", 0))
        coolant = float(obd.get("coolant_c", 0))
        oil_pressure = float(obd.get("oil_pressure_kpa", 0))
        battery = float(obd.get("battery_v", 0))
        speed = float(obd.get("speed_kmph", 0))

        features = np.array([[rpm, coolant, oil_pressure, battery, speed]])

        health_score = model.predict(features)[0]
        health_score = round(float(health_score), 2)

        print("VIN:", data.get("vin"))
        print("Predicted Health:", health_score)

        return jsonify({
            "status": "ok",
            "vehicle_health_score": health_score
        })

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# -------------------------------
# Health Check Route
# -------------------------------
@app.route("/")
def home():
    return "AutoMind ML Backend Running"


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)