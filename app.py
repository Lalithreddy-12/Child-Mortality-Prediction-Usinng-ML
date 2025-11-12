# app.py
import os
import json
import re
import time
import traceback
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient
import joblib
import pandas as pd
import logging

# --- Load environment variables ---
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
# If you prefer not to require HF, set this to False. For your request, we require HF.
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found in .env file")

# --- Logging ---
logger = logging.getLogger("child_mortality")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# --- Initialize Hugging Face client (best-effort) ---
try:
    client = InferenceClient(api_key=HF_API_KEY, timeout=120)
    logger.info("Initialized Hugging Face InferenceClient.")
except Exception as e:
    client = None
    logger.exception("Failed to initialize InferenceClient: %s", e)

# --- Flask app setup ---
APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "models", "model.pkl")
app = Flask(__name__, static_folder="static", static_url_path="/static")

# --- Load local ML model ---
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            logger.info("Loaded local model from %s", MODEL_PATH)
            return m
        except Exception as e:
            logger.exception("❌ Failed to load local model: %s", e)
    else:
        logger.warning("No local model found at %s", MODEL_PATH)
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

# --- Helper: parse JSON / free-text to structured plan ---
def _parse_plan_from_text(raw_text):
    expected_keys = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"]
    default_steps = list(generate_fallback_plan()["years"].values())

    if not raw_text or not isinstance(raw_text, str):
        return {k: default_steps[i] for i, k in enumerate(expected_keys)}

    # 1) Try to locate a JSON object block
    try:
        m = re.search(r"\{(?:[^{}]|\n|\r)*\}", raw_text, flags=re.S)
        if m:
            candidate = m.group(0)
            plan_json = json.loads(candidate)
            cleaned = {}
            for i, k in enumerate(expected_keys):
                v = plan_json.get(k)
                if isinstance(v, list) and len(v) > 0:
                    cleaned[k] = [str(x).strip() for x in v][:5]
                else:
                    cleaned[k] = default_steps[i]
            return cleaned
    except Exception:
        # fall through to heuristics
        pass

    # 2) Heuristic: look for headings like "Year 0-1" and bullets following them
    try:
        text = "\n".join([ln.strip() for ln in raw_text.splitlines() if ln.strip()])
        # Find headings and their positions
        heading_re = re.compile(r"(Year\s*\d+\s*(?:-|to|–)\s*\d+)", flags=re.I)
        headings = [(m.group(1).strip(), m.start()) for m in heading_re.finditer(text)]
        if headings:
            # build sections
            sections = {}
            positions = [pos for (_, pos) in headings] + [len(text)]
            keys = [h for (h, _) in headings]
            for idx, key in enumerate(keys):
                start = positions[idx]
                end = positions[idx + 1]
                seg = text[start:end].strip()
                # drop the heading line
                seg_lines = seg.splitlines()
                body = "\n".join(seg_lines[1:]) if len(seg_lines) > 1 else ""
                # split by bullets/newlines/commas/semicolons
                items = re.split(r"[\n\r]+|[•\-\*\u2022]+|[;,\|]\s*", body)
                items = [it.strip() for it in items if it and len(it.strip()) > 2]
                sections[key] = items

            # Map found sections to expected keys using digit matching
            cleaned = {}
            for i, ek in enumerate(expected_keys):
                ek_nums = re.findall(r"\d+", ek)
                found = None
                for pk, val in sections.items():
                    if re.findall(r"\d+", pk) == ek_nums and val:
                        found = val
                        break
                cleaned[ek] = found[:5] if found else default_steps[i]
            return cleaned
    except Exception:
        pass

    # 3) Fallback heuristic: split by "Year " token
    try:
        parts = re.split(r"(?:Year\s*\d+\s*(?:-|to|–)\s*\d+)", text, flags=re.I)
        hdrs = re.findall(r"(Year\s*\d+\s*(?:-|to|–)\s*\d+)", text, flags=re.I)
        heuristic = {}
        for i, hdr in enumerate(hdrs):
            try:
                body = parts[i + 1]
            except Exception:
                body = ""
            items = re.split(r"[\n\r]+|[•\-\*\u2022]+|[;,\|]\s*", body)
            items = [it.strip() for it in items if it and len(it.strip()) > 2]
            heuristic[hdr] = items
        cleaned = {}
        for i, ek in enumerate(expected_keys):
            ek_nums = re.findall(r"\d+", ek)
            found = None
            for pk, val in heuristic.items():
                if re.findall(r"\d+", pk) == ek_nums and val:
                    found = val
                    break
            cleaned[ek] = found[:5] if found else default_steps[i]
        return cleaned
    except Exception:
        pass

    # 4) Last resort: return defaults
    return {k: default_steps[i] for i, k in enumerate(expected_keys)}

# --- Cache wrapper to avoid repeated identical HF calls ---
@lru_cache(maxsize=256)
def _cached_generate_plan_tuple(features_tuple):
    # features_tuple must be a tuple of ordered feature values
    features = {
        "birth_weight": features_tuple[0],
        "maternal_age": features_tuple[1],
        "immunized": features_tuple[2],
        "nutrition": features_tuple[3],
        "socioeconomic": features_tuple[4],
        "prenatal_visits": features_tuple[5]
    }
    plan, debug = _generate_survival_plan_hf_uncached(features)
    # return JSON-serializable
    return json.dumps((plan, debug))

def generate_survival_plan_hf(features):
    """
    Public wrapper: uses LRU cache keyed by feature tuple, returns plan dict and debug dict.
    """
    # build deterministic tuple as cache key
    tup = (
        float(features.get("birth_weight", 0.0)),
        float(features.get("maternal_age", 0.0)),
        int(features.get("immunized", 0)),
        float(features.get("nutrition", 0.0)),
        int(features.get("socioeconomic", 0)),
        float(features.get("prenatal_visits", 0.0))
    )
    try:
        cached = _cached_generate_plan_tuple(tup)
        plan, debug = json.loads(cached)
        return plan, debug
    except Exception:
        # If cache decoding fails, call uncached function
        return _generate_survival_plan_hf_uncached({
            "birth_weight": tup[0],
            "maternal_age": tup[1],
            "immunized": tup[2],
            "nutrition": tup[3],
            "socioeconomic": tup[4],
            "prenatal_visits": tup[5]
        })

def _generate_survival_plan_hf_uncached(features, model_name=HF_MODEL, max_retries=1):
    """
    The robust HF caller (uncached). Returns (plan_dict, debug_dict).
    """
    if client is None:
        logger.warning("InferenceClient not initialized; returning fallback.")
        return generate_fallback_plan(), {"hf_raw": None, "error": "no_client"}

    baby_info = ", ".join([f"{k}: {v}" for k, v in features.items()])

    # Strong prompt with explicit JSON template
    system_prompt = (
        "You are a pediatric health assistant. PRODUCE EXACTLY ONE JSON OBJECT AND NOTHING ELSE.\n"
        "The JSON object must contain exactly these keys: "
        "'Year 0-1', 'Year 1-2', 'Year 2-3', 'Year 3-4', 'Year 4-5'.\n"
        "Each key's value must be an array (list) of 3 to 5 short actionable steps (strings).\n"
        "Do NOT include explanations, commentary, or any other text. Do NOT wrap output in Markdown or code fences.\n"
        "If you cannot produce the requested JSON, output a single empty JSON object: {}.\n\n"
        "EXAMPLE TEMPLATE (fill arrays with actual steps):\n"
        '{\n'
        '  "Year 0-1": ["step1", "step2", "step3"],\n'
        '  "Year 1-2": ["step1", "step2", "step3"],\n'
        '  "Year 2-3": ["step1", "step2", "step3"],\n'
        '  "Year 3-4": ["step1", "step2", "step3"],\n'
        '  "Year 4-5": ["step1", "step2", "step3"]\n'
        '}\n'
    )

    user_prompt = f"Child features: {baby_info}. Generate the JSON exactly as above."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    last_raw = None
    for attempt in range(1, max_retries + 2):
        try:
            logger.debug("Calling HF model=%s attempt=%d", model_name, attempt)
            start = time.time()
            resp = client.chat_completion(model=model_name, messages=messages, max_tokens=700)
            elapsed = time.time() - start

            # extract raw text
            try:
                raw_text = resp.choices[0].message["content"]
            except Exception:
                raw_text = str(resp)
            last_raw = (raw_text or "")[:8000]
            logger.debug("HF responded in %.2fs; raw-trunc=%s", elapsed, last_raw[:400].replace("\n", " "))

            # Parse into structured plan
            parsed = _parse_plan_from_text(raw_text)
            if parsed:
                return {"risk_level": "high", "years": parsed}, {"hf_raw": last_raw}
            else:
                # if parsed is empty, return fallback with debug
                return generate_fallback_plan(), {"hf_raw": last_raw, "error": "parsed_empty"}

        except Exception as exc:
            logger.exception("HF call attempt %d failed: %s", attempt, exc)
            last_raw = (str(exc) + "\n" + (last_raw or ""))[:8000]
            if attempt <= max_retries:
                time.sleep(2 ** attempt)
                continue
            return generate_fallback_plan(), {"hf_raw": last_raw, "error": str(exc)}

    return generate_fallback_plan(), {"hf_raw": last_raw, "error": "max_retries_exceeded"}

# --- Survival plan selector ---
def survival_plan(prediction, features=None, debug=False):
    if int(prediction) == 0:
        return {"risk_level": "low", "message": "Low risk. Continue preventive care: vaccines, nutrition, safe environment."}, {"branch": "low"}
    else:
        if features:
            plan, debug_info = generate_survival_plan_hf(features)
            return plan, {"branch": "high", **(debug_info or {})}
        else:
            return generate_fallback_plan(), {"branch": "high", "hf_raw": None, "error": "no_features"}

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

    # check debug toggle: query param ?debug=true or body { "debug": true }
    q_debug = request.args.get("debug", "").lower() == "true"
    body_debug = bool(data.get("debug")) if isinstance(data, dict) else False
    debug_flag = q_debug or body_debug

    try:
        df = _build_input_df(data)
    except ValueError as e:
        return jsonify({"error": "Invalid input format", "details": str(e)}), 400

    prob = float(model.predict_proba(df)[:, 1][0]) if hasattr(model, "predict_proba") else None
    pred = int(model.predict(df)[0])
    features = df.iloc[0].to_dict()

    plan_obj, debug_info = survival_plan(pred, features, debug=debug_flag)

    response = {
        "mortality_risk_probability": prob,
        "mortality_prediction": pred,
        "interpretation": "1 means higher predicted risk, 0 means lower predicted risk",
        "survival_plan": plan_obj,
        "debug": {"branch": debug_info.get("branch", "high" if pred == 1 else "low")}
    }

    if debug_flag:
        hf_raw = debug_info.get("hf_raw")
        if hf_raw:
            response["debug"]["hf_raw_truncated"] = hf_raw[:3000]
        if "error" in debug_info:
            response["debug"]["error"] = debug_info["error"]

    return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict_alias():
    return predict_api()

# --- Run server ---
if __name__ == "__main__":
    print("🚀 Starting Child Mortality API server on http://127.0.0.1:8000 ")
    app.run(host="0.0.0.0", port=8000, debug=True)
