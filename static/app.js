// static/app.js
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const predictBtn = document.getElementById("predict");
  const clearBtn = document.getElementById("clear");
  const resultCard = document.getElementById("resultCard");
  const planContainer = document.getElementById("planContainer");
  const meterCircle = document.getElementById("meterCircle");
  const riskLabel = document.getElementById("riskLabel");
  const probLabel = document.getElementById("probLabel");
  const probBar = document.getElementById("probBar");
  const explainBtn = document.getElementById("explainBtn");
  const explainText = document.getElementById("explainText");
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");

  // Theme: persistent toggle
  const applyTheme = (dark) => {
    if (dark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    themeIcon.textContent = dark ? "☀️" : "🌙";
    localStorage.setItem("dm_theme", dark ? "dark" : "light");
  };
  applyTheme(localStorage.getItem("dm_theme") === "dark");
  themeToggle.addEventListener("click", () => applyTheme(!(localStorage.getItem("dm_theme") === "dark")));

  // Helpers for meter circle (SVG arc uses stroke-dasharray)
  function setMeter(percent) {
    // percent: 0-100
    const circumference = 100; // using stroke-dasharray as percentage of 100
    const dash = (percent / 100) * circumference;
    // Color by risk
    let color = "#10b981"; // green
    if (percent >= 70) color = "#ef4444"; // red
    else if (percent >= 40) color = "#f59e0b"; // orange
    meterCircle.style.stroke = color;
    meterCircle.setAttribute("stroke-dasharray", `${dash}, ${circumference}`);
  }

  // Animate adding a plan card
  async function showPlan(planObj) {
    planContainer.innerHTML = "";
    const years = planObj?.years || {};
    const expectedYears = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"];
    let delay = 0;
    for (const year of expectedYears) {
      const actions = Array.isArray(years[year]) && years[year].length ? years[year] : ["No data available"];
      const card = document.createElement("div");
      card.className = "plan-card bg-white dark:bg-gray-800 p-4 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 animate-fade-up";
      card.innerHTML = `
        <h4 class="font-semibold text-primary">${year}</h4>
        <ul class="mt-2 text-sm text-gray-700 dark:text-gray-200 space-y-1">
          ${actions.map(a => `<li>• ${a}</li>`).join('')}
        </ul>
      `;
      planContainer.appendChild(card);
      // stagger in
      setTimeout(() => card.classList.add("in"), 80 + delay);
      delay += 120;
    }
  }

  // Clear handler
  clearBtn.addEventListener("click", () => {
    document.getElementById("birth_weight").value = "2.8";
    document.getElementById("maternal_age").value = "26";
    document.getElementById("immunized").value = "1";
    document.getElementById("nutrition").value = "60";
    document.getElementById("socioeconomic").value = "1";
    document.getElementById("prenatal_visits").value = "4";
    resultCard.classList.add("hidden");
    explainText.classList.add("hidden");
    explainText.innerHTML = "";
    planContainer.innerHTML = "";
  });

  // Explain button (populates from server debug if available)
  explainBtn.addEventListener("click", () => {
    if (explainText.classList.contains("hidden")) {
      // show explanation if available (stored in element dataset)
      explainText.innerHTML = explainText.dataset.text || "Top features not available for this demo.";
      explainText.classList.remove("hidden");
    } else {
      explainText.classList.add("hidden");
    }
  });

  // Main predict handler
  predictBtn.addEventListener("click", async () => {
    // gather inputs
    const payload = {
      birth_weight: parseFloat(document.getElementById("birth_weight").value || 0),
      maternal_age: parseFloat(document.getElementById("maternal_age").value || 0),
      immunized: parseInt(document.getElementById("immunized").value || 0),
      nutrition: parseFloat(document.getElementById("nutrition").value || 0),
      socioeconomic: parseInt(document.getElementById("socioeconomic").value || 0),
      prenatal_visits: parseFloat(document.getElementById("prenatal_visits").value || 0),
      debug: true
    };

    predictBtn.disabled = true;
    predictBtn.classList.add("opacity-70");
    predictBtn.innerHTML = `<svg class="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="white" stroke-width="4" fill="none"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path></svg> Processing`;

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });

      if (!resp.ok) throw new Error(`Server error ${resp.status}`);

      const data = await resp.json();
      // show card
      resultCard.classList.remove("hidden");

      const prob = (data.mortality_risk_probability != null) ? Number(data.mortality_risk_probability) * 100 : null;
      const pred = data.mortality_prediction;
      const riskText = (pred === 1 || (prob !== null && prob >= 60)) ? "High" : (prob !== null && prob >= 35 ? "Medium" : "Low");

      // set meter/prob
      setMeter(prob || 0);
      riskLabel.textContent = `Risk: ${riskText}`;
      probLabel.textContent = `Probability: ${prob !== null ? prob.toFixed(1) + "%" : "N/A"}`;
      probBar.style.width = `${prob !== null ? Math.min(100, prob) : 0}%`;
      probBar.style.backgroundColor = (prob >= 70 ? "#ef4444" : prob >= 40 ? "#f59e0b" : "#10b981");

      // explain text (if server returned debug/hf info with top features)
      const hfRaw = data.debug?.hf_raw_truncated || null;
      if (data.explain || data.explanation) {
        // if you later add explanation to response, show it
        explainText.dataset.text = data.explain || data.explanation;
      } else if (hfRaw) {
        // fallback: show short snippet of HF raw text as explanation context
        explainText.dataset.text = hfRaw.slice(0, 600) + (hfRaw.length > 600 ? "…" : "");
      } else {
        explainText.dataset.text = "Top contributing features not available for this demo.";
      }
      explainText.classList.add("hidden");

      // animate plan blocks
      await showPlan(data.survival_plan || { years: {} });

      // smooth scroll to results on small screens
      resultCard.scrollIntoView({behavior: "smooth", block: "center"});

    } catch (err) {
      console.error(err);
      resultCard.classList.remove("hidden");
      planContainer.innerHTML = `<div class="col-span-1 p-4 text-sm text-red-600">Error: ${err.message}</div>`;
      explainText.dataset.text = "";
    } finally {
      predictBtn.disabled = false;
      predictBtn.classList.remove("opacity-70");
      predictBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10 5v10m5-5H5"/></svg> Predict`;
    }
  });
});
