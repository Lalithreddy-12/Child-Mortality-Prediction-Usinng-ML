from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import numpy as np
import pandas as pd

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

# ✅ Survival plan recommender
def survival_plan(prediction, features=None):
    """
    Returns a JSON-serializable plan object.
    For low risk -> {'message': '...'}
    For high risk -> {'years': { 'Year 0-1': [...], ... }}
    """
    if int(prediction) == 0:
        return {
            "risk_level": "low",
            "message": "Low risk. Continue preventive care: vaccines, nutrition, safe environment."
        }
    else:
        return {
            "risk_level": "high",
            "years": {
                "Year 0-1": [
                    "Exclusive breastfeeding",
                    "Routine vaccinations (per national schedule)",
                    "Monthly pediatric check-ups",
                    "Safe sleep & clean water"
                ],
                "Year 1-2": [
                    "Introduce diverse solid foods (iron-rich)",
                    "Measles/MMR vaccination as scheduled",
                    "Monitor growth & anemia screening",
                    "Oral rehydration plan for diarrhea"
                ],
                "Year 2-3": [
                    "Vitamin A supplementation if recommended",
                    "Safe play environment & injury prevention",
                    "Screen for malnutrition & deworming if indicated",
                    "Hygiene: handwashing & safe sanitation"
                ],
                "Year 3-4": [
                    "Preschool enrolment / early stimulation",
                    "Annual vision & dental checkups",
                    "Sleep routine, physical activity",
                    "Update any missed vaccines"
                ],
                "Year 4-5": [
                    "Booster immunizations (per schedule)",
                    "Balanced diet with regular growth monitoring",
                    "School readiness support & mental wellbeing",
                    "Emergency care plan (fever, respiratory issues)"
                ]
            }
        }

@app.route("/api/predict", methods=["POST"])
def predict():
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

    # Probability (if model supports it)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[:, 1][0])

    pred = int(model.predict(df)[0])
    plan_obj = survival_plan(pred, df.iloc[0].to_dict())

    response = {
        "mortality_risk_probability": prob,
        "mortality_prediction": pred,
        "interpretation": "1 means higher predicted risk, 0 means lower predicted risk",
        # Always include survival_plan to make frontend simpler
        "survival_plan": plan_obj,
        "debug": {"branch": "high" if pred == 1 else "low"}
    }
    return jsonify(response)

@app.route("/api/accuracy", methods=["GET"])
def accuracy():
    from sklearn.metrics import accuracy_score
    try:
        df = pd.read_csv("models/sample_input.csv")  # small sample or replace with your dataset
        X = df.drop(columns=["mortality"])
        y_true = df["mortality"]
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        return jsonify({
            "accuracy": float(acc),
            "tested_samples": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/explain", methods=["POST"])
def explain():
    global model
    if model is None:
        return jsonify({"error": "Model not found. Please run train_model.py"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        df = _build_input_df(data)
    except ValueError as e:
        return jsonify({"error": "Invalid input format", "details": str(e)}), 400

    # ✅ SHAP explainability (inside function, with robust handling)
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        # Handle binary vs multi-class vs single array case
        if isinstance(shap_values, list):
            # Old style: list per class → pick class 1 if available
            shap_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap_vals = shap_class[0]  # take first sample
        else:
            # New style: already array
            shap_vals = shap_values[0]

        # Convert to numpy safely
        if hasattr(shap_vals, "values"):  # sometimes it's a shap.Explanation
            vals = np.array(shap_vals.values)
        else:
            vals = np.array(shap_vals)

        # Ensure 1D row vector (first sample)
        if vals.ndim > 1:
            vals = vals[0]

        feature_contribs = {col: float(val) for col, val in zip(df.columns, vals)}

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[:, 1][0])

        pred = int(model.predict(df)[0])
        plan_obj = survival_plan(pred, df.iloc[0].to_dict())

        return jsonify({
            "features": feature_contribs,
            "prediction": pred,
            "probability": probability,
            "interpretation": "Positive SHAP value = pushes towards higher risk",
            # Also include the survival plan here
            "survival_plan": plan_obj,
            "debug": {"branch": "high" if pred == 1 else "low"}
        })

    except Exception as e:
        return jsonify({"error": "SHAP failed", "details": str(e)}), 500


if __name__ == "__main__":
    # If you’re using a frontend on a different origin, consider enabling CORS.
    app.run(host="0.0.0.0", port=8000, debug=True)
