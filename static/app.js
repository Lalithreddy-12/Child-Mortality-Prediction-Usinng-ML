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
      let html = `
        <strong>Predicted mortality risk probability:</strong> ${j.mortality_risk_probability.toFixed(3)}<br>
        <strong>Prediction (1 = higher risk):</strong> ${j.mortality_prediction}
      `;

      // Handle survival plan in flexible ways
      if (j.survival_plan) {
        if (typeof j.survival_plan === "string") {
          // Simple text recommendation
          html += `<br><br><strong>Recommendation:</strong> ${j.survival_plan}`;
        } else if (typeof j.survival_plan === "object") {
          if (j.survival_plan.risk_level === "low") {
            html += `<br><br><strong>Recommendation:</strong> ${j.survival_plan.message}`;
          } else if (j.survival_plan.risk_level === "high" && j.survival_plan.years) {
            html += "<br><br><strong>5-Year Survival Plan:</strong><ul>";
            for (const [year, actions] of Object.entries(j.survival_plan.years)) {
              html += `<li><b>${year}</b>: ${actions.join(", ")}</li>`;
            }
            html += "</ul>";
          } else {
            // fallback if it’s just year: actions mapping
            html += "<br><br><strong>5-Year Survival Plan:</strong><ul>";
            for (const [year, actions] of Object.entries(j.survival_plan)) {
              html += `<li><b>${year}</b>: ${actions.join(", ")}</li>`;
            }
            html += "</ul>";
          }
        }
      }

      resEl.innerHTML = html;
    } else {
      resEl.innerHTML = "<strong>Error:</strong> " + (j.error || JSON.stringify(j));
    }
  } catch (e) {
    resEl.innerHTML = "<strong>Network error:</strong> " + e;
  }
});
