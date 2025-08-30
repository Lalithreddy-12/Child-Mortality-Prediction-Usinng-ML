import requests
import json

# URL of your Flask backend (make sure app.py is running)
URL = "http://127.0.0.1:8000/api/explain"

# Example input (you can change values for testing)
payload = {
    "birth_weight": 2.5,
    "maternal_age": 24,
    "immunized": 0,
    "nutrition": 40,
    "socioeconomic": 0,
    "prenatal_visits": 2
}

try:
    response = requests.post(URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Explanation Result from /api/explain\n")
        print("Prediction (1 = higher risk):", data.get("prediction"))
        print("Predicted probability of mortality risk:", round(data.get("probability", 0), 3))
        print("\nFeature Contributions (SHAP values):")
        for feature, value in data.get("features", {}).items():
            print(f"  {feature}: {value:+.4f}")
        print("\nInterpretation:", data.get("interpretation"))
    else:
        print("❌ Error:", response.status_code, response.text)
except Exception as e:
    print("⚠️ Request failed:", str(e))
