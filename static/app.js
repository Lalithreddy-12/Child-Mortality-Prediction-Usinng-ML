document.getElementById("predict").addEventListener("click", async () => {
  const payload = {
    birth_weight: parseFloat(document.getElementById("birth_weight").value),
    maternal_age: parseFloat(document.getElementById("maternal_age").value),
    immunized: parseInt(document.getElementById("immunized").value),
    nutrition: parseFloat(document.getElementById("nutrition").value),
    socioeconomic: parseInt(document.getElementById("socioeconomic").value),
    prenatal_visits: parseFloat(document.getElementById("prenatal_visits").value)
  };
  const resEl = document.getElementById("result");
  resEl.style.display = "block";
  resEl.textContent = "Contacting server...";
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (r.ok) {
      resEl.innerHTML = "<strong>Predicted mortality risk probability:</strong> " + j.mortality_risk_probability.toFixed(3) + "<br><strong>Prediction (1 = higher risk):</strong> " + j.mortality_prediction;
    } else {
      resEl.innerHTML = "<strong>Error:</strong> " + (j.error || JSON.stringify(j));
    }
  } catch (e) {
    resEl.innerHTML = "<strong>Network error:</strong> " + e;
  }
});