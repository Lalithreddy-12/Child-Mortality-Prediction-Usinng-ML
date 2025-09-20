document.addEventListener("DOMContentLoaded", function () {
  const predictBtn = document.getElementById("predict");
  const resultDiv = document.getElementById("result");

  predictBtn.addEventListener("click", async function () {
    const birth_weight = parseFloat(document.getElementById("birth_weight").value);
    const maternal_age = parseFloat(document.getElementById("maternal_age").value);
    const immunized = parseInt(document.getElementById("immunized").value);
    const nutrition = parseFloat(document.getElementById("nutrition").value);
    const socioeconomic = parseInt(document.getElementById("socioeconomic").value);
    const prenatal_visits = parseInt(document.getElementById("prenatal_visits").value);

    const payload = { birth_weight, maternal_age, immunized, nutrition, socioeconomic, prenatal_visits };

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Response received:", data);

      // Build survival plan HTML safely
      const years = data.survival_plan?.years || {};
      const expectedYears = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"];
      let survivalPlanHtml = "<h3>Survival Plan</h3>";

      expectedYears.forEach(year => {
        const actions = Array.isArray(years[year]) && years[year].length > 0 ? years[year] : ["No data available"];
        survivalPlanHtml += `<p><b>${year} years:</b></p><ul>`;
        actions.forEach(action => {
          survivalPlanHtml += `<li>${action}</li>`;
        });
        survivalPlanHtml += "</ul>";
      });

      resultDiv.style.display = "block";
      resultDiv.innerHTML = `
        <p><b>Mortality Prediction:</b> ${data.mortality_prediction ?? "N/A"}</p>
        <p><b>Probability:</b> ${data.mortality_risk_probability != null ? data.mortality_risk_probability.toFixed(3) : "N/A"}</p>
        <p><b>Risk Level:</b> ${data.survival_plan?.risk_level || "N/A"}</p>
        ${survivalPlanHtml}
      `;

    } catch (error) {
      console.error("Error:", error);
      resultDiv.style.display = "block";
      resultDiv.innerHTML = `<span style="color:red">Error: ${error.message}</span>`;
    }
  });
});
