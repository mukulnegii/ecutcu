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

    data = request.json
    obd = data.get("obd", {})

    try:
        speed = float(obd.get("speed_kmph", 0))
        rpm = float(obd.get("rpm", 0))
        engine_temp = float(obd.get("coolant_c", 0))
        brake_pressure = float(obd.get("oil_pressure_kpa", 0))  # substitute
        battery_voltage = float(obd.get("battery_v", 0))

        gear_str = obd.get("gear_position", "P")

        # Encode gear (must match training encoding)
        gear_map = {"P": 0, "N": 1, "R": 2, "D": 3}
        gear = gear_map.get(gear_str, 0)

        features = [[
            speed,
            rpm,
            engine_temp,
            brake_pressure,
            battery_voltage,
            gear
        ]]

        health_score = model.predict(features)[0]
        health_score = round(float(health_score), 2)

        print("VIN:", data.get("vin"))
        print("Predicted Health:", health_score)

        return jsonify({
            "status": "ok",
            "vehicle_health_score": health_score
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


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