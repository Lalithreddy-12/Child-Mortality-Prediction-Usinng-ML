from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import json
import re
import traceback

# --- Load environment variables ---
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# --- Hugging Face client (GPT-like model) ---
hf_client = None
if HF_API_KEY:
    try:
        hf_client = InferenceClient(api_key=HF_API_KEY, timeout=120)
    except Exception as e:
        print("⚠️ Hugging Face client init failed:", e)

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "models", "model.pkl")

app = Flask(__name__, static_folder="static", static_url_path="/static")

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

def _build_input_df(data):
    """Builds a 1-row DataFrame from incoming JSON. Raises ValueError on bad input."""
    try:
        return pd.DataFrame([{
            "birth_weight": float(data.get("birth_weight")),
            "maternal_age": float(data.get("maternal_age")),
            "immunized": int(data.get("immunized")),
            "nutrition": float(data.get("nutrition")),
            "socioeconomic": int(data.get("socioeconomic")),
            "prenatal_visits": float(data.get("prenatal_visits"))
        }])
    except Exception as e:
        raise ValueError(f"Invalid input format or missing field: {e}")

# --- GPT-powered survival plan generator ---
def generate_survival_plan_gpt(features):
    if not hf_client:
        return {"error": "Hugging Face client not initialized. Check HF_API_KEY."}

    baby_info = ", ".join([f"{k}: {v}" for k, v in features.items()])
    messages = [
        {"role": "system", "content": "You are a medical AI assistant. Return ONLY valid JSON with keys: risk_level (string) and years (dict of year: [recommendations])."},
        {"role": "user", "content": f"Based on this data: {baby_info}, generate the JSON survival plan."}
    ]

    try:
        response = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )

        raw = response.choices[0].message["content"].strip()
        #print("\n🤖 Raw HuggingFace response:\n", raw, "\n")  # debug log

        # --- Clean JSON fences ---
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
            raw = re.sub(r"```$", "", raw).strip()

        # Extract JSON only
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError(f"No JSON detected. Got: {raw[:200]}...")

        json_str = match.group(0)

        # --- Attempt strict parse ---
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # 🔧 Relax parsing: remove trailing commas
            cleaned = re.sub(r",\s*([\]}])", r"\1", json_str)
            parsed = json.loads(cleaned)

        # Ensure years is dict of lists
        fixed_years = {}
        for k, v in parsed.get("years", {}).items():
            if isinstance(v, list):
                fixed_years[k] = v
            else:
                fixed_years[k] = [v]

        return {
            "risk_level": parsed.get("risk_level", "unknown"),
            "years": fixed_years
        }

    except Exception as e:
        print("❌ GPT survival plan error:", str(e))
        traceback.print_exc()

        return {
            "risk_level": "high",
            "years": {
                "Year 0-1": ["Doctor visits", "Vaccinations"],
                "Year 1-2": ["Growth monitoring", "Nutrition check"],
                "Year 2-3": ["Speech & motor skills"],
                "Year 3-4": ["School readiness"],
                "Year 4-5": ["Annual screening"]
            },
            "warning": f"Used fallback due to parsing error: {e}"
        }

# ✅ Survival plan selector
def survival_plan(prediction, features=None):
    if int(prediction) == 0:
        return {
            "risk_level": "low",
            "message": "Low risk. Continue preventive care: vaccines, nutrition, safe environment."
        }
    else:
        if features:
            return generate_survival_plan_gpt(features)
        else:
            return {"risk_level": "high", "message": "No features available for GPT generation."}

@app.route("/api/predict", methods=["POST"])
def predict_api():
    global model
    if model is None:
        return jsonify({"error": "Model not found. Please run train_model.py to create models/model.pkl"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        df = _build_input_df(data)
    except ValueError as e:
        return jsonify({"error": "Invalid input format", "details": str(e)}), 400

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[:, 1][0])

    pred = int(model.predict(df)[0])
    features = df.iloc[0].to_dict()
    #print("🍼 User features:", features)  # debug log

    plan_obj = survival_plan(pred, features)

    response = {
        "mortality_risk_probability": prob,
        "mortality_prediction": pred,
        "interpretation": "1 means higher predicted risk, 0 means lower predicted risk",
        "survival_plan": plan_obj,
        "debug": {"branch": "high" if pred == 1 else "low"}
    }
    return jsonify(response)

# ✅ New wrapper so frontend calling `/predict` works too
@app.route("/predict", methods=["POST"])
def predict_alias():
    return predict_api()

# (accuracy & explain endpoints remain unchanged)...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
