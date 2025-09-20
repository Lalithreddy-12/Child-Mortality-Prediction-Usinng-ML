import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient
import joblib

# --- Load environment variables ---
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found in .env file")

# --- Initialize Hugging Face client ---
client = InferenceClient(api_key=HF_API_KEY, timeout=120)

# --- Flask app setup ---
APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "models", "model.pkl")
app = Flask(__name__, static_folder="static", static_url_path="/static")

# --- Load local ML model ---
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print("❌ Failed to load local model:", e)
    return None

model = load_model()

# --- Build input DataFrame ---
def _build_input_df(data):
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

# --- Fallback survival plan ---
def generate_fallback_plan():
    return {
        "risk_level": "high",
        "years": {
            "Year 0-1": ["Doctor visits", "Vaccinations", "Monitor growth and nutrition"],
            "Year 1-2": ["Regular checkups", "Balanced diet", "Monitor development milestones"],
            "Year 2-3": ["Speech and motor skill support", "Vaccinations update", "Nutritional supplements if needed"],
            "Year 3-4": ["School readiness assessment", "Preventive health checks", "Encourage physical activity"],
            "Year 4-5": ["Annual pediatric screening", "Vaccinations", "Healthy lifestyle counseling"]
        },
        "warning": "Used fallback plan due to HF API failure or parsing error."
    }

# --- HF chat-based survival plan with guaranteed non-empty years ---
def generate_survival_plan_hf(features):
    if not HF_API_KEY:
        return generate_fallback_plan()

    baby_info = ", ".join([f"{k}: {v}" for k, v in features.items()])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a world-class pediatric healthcare advisor. "
                "Provide a 5-year survival plan ONLY in JSON format. "
                "Keys: 'Year 0-1', 'Year 1-2', 'Year 2-3', 'Year 3-4', 'Year 4-5'. "
                "Each key must be an array of 3–5 actionable steps. "
                "Do NOT include explanations, calculations, or extra text. "
                "If uncertain, output placeholder steps."
            )
        },
        {"role": "user", "content": f"Generate a 5-year survival plan for a child with these features: {baby_info}"}
    ]

    try:
        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            max_tokens=500
        )

        raw_text = response.choices[0].message["content"]

        # --- Extract JSON if possible ---
        plan_json = {}
        try:
            match = re.search(r"\{.*\}", raw_text, flags=re.S)
            if match:
                plan_json = json.loads(match.group(0))
        except:
            pass

        # --- Ensure each year has meaningful steps ---
        expected_keys = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"]
        default_steps = [
            ["Doctor visits", "Vaccinations", "Monitor growth and nutrition"],
            ["Checkups", "Balanced diet", "Development monitoring"],
            ["Speech/motor support", "Vaccination updates", "Nutritional supplements"],
            ["School readiness", "Preventive health checks", "Physical activity encouragement"],
            ["Annual pediatric screening", "Vaccinations", "Healthy lifestyle guidance"]
        ]

        cleaned_plan = {}
        for i, key in enumerate(expected_keys):
            val = plan_json.get(key)
            if isinstance(val, list) and len(val) > 0:
                cleaned_plan[key] = val
            else:
                # Use default steps if model output is missing or invalid
                cleaned_plan[key] = default_steps[i]

        return {"risk_level": "high", "years": cleaned_plan}

    except Exception as e:
        print("❌ HF API Error:", e)
        return generate_fallback_plan()

# --- Survival plan selector ---
def survival_plan(prediction, features=None):
    if int(prediction) == 0:
        return {
            "risk_level": "low",
            "message": "Low risk. Continue preventive care: vaccines, nutrition, safe environment."
        }
    else:
        if features and HF_API_KEY:
            return generate_survival_plan_hf(features)
        else:
            return generate_fallback_plan()

# --- Flask routes ---
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/predict", methods=["POST"])
def predict_api():
    global model
    if model is None:
        return jsonify({"error": "Local model not found. Please train it or rely on HF_API_KEY."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        df = _build_input_df(data)
    except ValueError as e:
        return jsonify({"error": "Invalid input format", "details": str(e)}), 400

    prob = float(model.predict_proba(df)[:, 1][0]) if hasattr(model, "predict_proba") else None
    pred = int(model.predict(df)[0])
    features = df.iloc[0].to_dict()

    plan_obj = survival_plan(pred, features)

    return jsonify({
        "mortality_risk_probability": prob,
        "mortality_prediction": pred,
        "interpretation": "1 means higher predicted risk, 0 means lower predicted risk",
        "survival_plan": plan_obj,
        "debug": {"branch": "high" if pred == 1 else "low"}
    })

@app.route("/predict", methods=["POST"])
def predict_alias():
    return predict_api()

# --- Run server ---
if __name__ == "__main__":
    print("🚀 Starting Child Mortality API server on http://0.0.0.0:8000 ...")
    app.run(host="0.0.0.0", port=8000, debug=True)
