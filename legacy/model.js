const defaults = {
  horizon: 20,
  performanceSpeed: 0.18,
  demandElasticity: 0.55,
  friction: 0.38,
  capitalBottleneck: 0.22,
  policyCushion: 0.3,
  networkEffect: 0.28,
  tippingSharpness: 0.18,
};

let params = { ...defaults };

let sectors = [
  { name: "Professional services", jobs: 23, exposure: 0.62, substitution: 0.42, complementarity: 0.28, threshold: 0.35, demand: 0.55, wage: 0.72 },
  { name: "Finance and insurance", jobs: 7.5, exposure: 0.70, substitution: 0.46, complementarity: 0.18, threshold: 0.32, demand: 0.45, wage: 0.80 },
  { name: "Manufacturing", jobs: 13, exposure: 0.38, substitution: 0.30, complementarity: 0.20, threshold: 0.48, demand: 0.70, wage: 0.50 },
  { name: "Healthcare", jobs: 22, exposure: 0.36, substitution: 0.18, complementarity: 0.36, threshold: 0.55, demand: 0.85, wage: 0.42 },
  { name: "Education", jobs: 9, exposure: 0.48, substitution: 0.24, complementarity: 0.42, threshold: 0.50, demand: 0.65, wage: 0.38 },
  { name: "Retail and hospitality", jobs: 25, exposure: 0.32, substitution: 0.26, complementarity: 0.14, threshold: 0.44, demand: 0.80, wage: 0.45 },
  { name: "Transport and logistics", jobs: 10, exposure: 0.30, substitution: 0.25, complementarity: 0.12, threshold: 0.58, demand: 0.72, wage: 0.52 },
  { name: "Public administration", jobs: 8, exposure: 0.54, substitution: 0.32, complementarity: 0.22, threshold: 0.45, demand: 0.35, wage: 0.35 },
];

const els = {
  table: document.getElementById("sectorTable"),
  chart: document.getElementById("chart"),
  bars: document.getElementById("sectorBars"),
  employmentChange: document.getElementById("employmentChange"),
  peakUnemployment: document.getElementById("peakUnemployment"),
  outputGain: document.getElementById("outputGain"),
  thresholdYear: document.getElementById("thresholdYear"),
};

const outputs = {
  horizon: document.getElementById("horizonOut"),
  performanceSpeed: document.getElementById("speedOut"),
  demandElasticity: document.getElementById("demandOut"),
  friction: document.getElementById("frictionOut"),
  capitalBottleneck: document.getElementById("bottleneckOut"),
  policyCushion: document.getElementById("policyOut"),
  networkEffect: document.getElementById("networkOut"),
  tippingSharpness: document.getElementById("tippingOut"),
};

const colors = {
  employment: "#24735f",
  output: "#2c63b7",
  wage: "#a87924",
  exposed: "#6954a3",
  grid: "#c5ccd7",
  text: "#5b6573",
};

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, Number(value)));
}

function sigmoid(x, sharpness) {
  return 1 / (1 + Math.exp(-x / Math.max(sharpness, 0.01)));
}

function formatPct(value, decimals = 1) {
  return `${(value * 100).toFixed(decimals)}%`;
}

function formatSignedPct(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)}%`;
}

function renderOutputs() {
  outputs.horizon.value = params.horizon;
  outputs.performanceSpeed.value = Math.round(params.performanceSpeed * 100);
  outputs.demandElasticity.value = params.demandElasticity.toFixed(2);
  outputs.friction.value = params.friction.toFixed(2);
  outputs.capitalBottleneck.value = params.capitalBottleneck.toFixed(2);
  outputs.policyCushion.value = params.policyCushion.toFixed(2);
  outputs.networkEffect.value = params.networkEffect.toFixed(2);
  outputs.tippingSharpness.value = params.tippingSharpness.toFixed(2);
}

function renderTable() {
  els.table.innerHTML = "";
  sectors.forEach((sector, index) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><input data-index="${index}" data-field="name" type="text" value="${sector.name}" aria-label="Sector name"></td>
      <td><input data-index="${index}" data-field="jobs" type="number" min="0" step="0.1" value="${sector.jobs}" aria-label="Jobs in millions"></td>
      <td><input data-index="${index}" data-field="exposure" type="number" min="0" max="1" step="0.01" value="${sector.exposure}" aria-label="Task exposure"></td>
      <td><input data-index="${index}" data-field="substitution" type="number" min="0" max="1" step="0.01" value="${sector.substitution}" aria-label="Substitution share"></td>
      <td><input data-index="${index}" data-field="complementarity" type="number" min="0" max="1" step="0.01" value="${sector.complementarity}" aria-label="Complementarity"></td>
      <td><input data-index="${index}" data-field="threshold" type="number" min="0" max="1" step="0.01" value="${sector.threshold}" aria-label="Adoption threshold"></td>
      <td><input data-index="${index}" data-field="demand" type="number" min="0" max="2" step="0.01" value="${sector.demand}" aria-label="Sector demand elasticity"></td>
      <td><input data-index="${index}" data-field="wage" type="number" min="0" max="2" step="0.01" value="${sector.wage}" aria-label="Wage sensitivity"></td>
      <td><button class="delete-btn" data-delete="${index}" type="button" aria-label="Delete sector">x</button></td>
    `;
    els.table.appendChild(row);
  });
}

function simulate() {
  const baseJobs = sectors.reduce((sum, sector) => sum + Number(sector.jobs), 0);
  const baseOutput = sectors.reduce((sum, sector) => sum + Number(sector.jobs) * (1 + sector.complementarity), 0);
  const rows = [];
  const sectorEnd = [];
  let previousAdoption = sectors.map(() => 0);
  let unemploymentStock = 0;
  let wageIndex = 1;

  for (let year = 0; year <= params.horizon; year += 1) {
    const frontier = Math.exp(params.performanceSpeed * year);
    const capability = 1 - Math.exp(-0.34 * (frontier - 1));
    const avgAdoption = previousAdoption.reduce((sum, value) => sum + value, 0) / previousAdoption.length;
    let employment = 0;
    let output = 0;
    let automatedTasks = 0;
    let displacementPressure = 0;
    let wagePressure = 0;
    const adoptionNow = [];
    const sectorRows = [];

    sectors.forEach((sector, index) => {
      const taskReadiness = sigmoid(capability - sector.threshold, params.tippingSharpness);
      const automatable = sector.exposure * sector.substitution * taskReadiness;
      const productivity = automatable * adoptionMultiplier(frontier) + sector.complementarity * capability * 0.42;
      const roi = productivity + params.networkEffect * avgAdoption - params.capitalBottleneck * (1 - taskReadiness);
      const adoption = sigmoid(roi - sector.threshold * 0.55, params.tippingSharpness);
      const adoptedAutomatable = automatable * adoption;
      const sectorDemand = 1 + 0.012 * year + (params.demandElasticity + sector.demand) * productivity * 0.34;
      const complementJobs = sector.complementarity * adoption * capability * 0.28;
      const laborPerOutput = Math.max(0.03, (1 - adoptedAutomatable + complementJobs) / (1 + productivity * 0.9));
      const desiredJobs = sector.jobs * sectorDemand * laborPerOutput;
      const transitionDrag = Math.max(0, sector.jobs - desiredJobs) * (params.friction * (1 - params.policyCushion));
      const sectorEmployment = Math.max(0, desiredJobs - transitionDrag);
      const sectorOutput = sector.jobs * (1 + productivity) * sectorDemand;
      const displacement = Math.max(0, sector.jobs - sectorEmployment);
      const sectorWagePressure = (productivity * 0.45 + complementJobs - adoptedAutomatable * sector.wage) * (sector.jobs / baseJobs);

      employment += sectorEmployment;
      output += sectorOutput;
      automatedTasks += adoptedAutomatable * sector.jobs;
      displacementPressure += displacement;
      wagePressure += sectorWagePressure;
      adoptionNow[index] = adoption;
      sectorRows.push({
        name: sector.name,
        employment: sectorEmployment,
        change: (sectorEmployment - sector.jobs) / Math.max(sector.jobs, 0.01),
        adoption,
        automated: adoptedAutomatable,
        displacement,
      });
    });

    unemploymentStock = unemploymentStock * (0.64 - params.policyCushion * 0.22) + displacementPressure * params.friction * 0.18;
    wageIndex = Math.max(0.25, wageIndex * (1 + wagePressure * 0.18 - unemploymentStock / baseJobs * 0.08));
    const instability = displacementPressure / baseJobs + unemploymentStock / baseJobs + Math.max(0, avgAdoption - 0.65) * params.friction;

    rows.push({
      year,
      capability,
      employment: employment / baseJobs,
      output: output / baseOutput,
      unemployment: unemploymentStock / baseJobs,
      wage: wageIndex,
      automatedTasks: automatedTasks / baseJobs,
      instability,
      sectors: sectorRows,
    });

    if (year === params.horizon) {
      sectorEnd.push(...sectorRows);
    }
    previousAdoption = adoptionNow;
  }

  return { rows, sectorEnd, baseJobs };
}

function adoptionMultiplier(frontier) {
  return Math.min(2.4, Math.log(frontier + 0.35) * 0.62);
}

function drawChart(rows) {
  const canvas = els.chart;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(Math.max(360, rect.width * 0.44) * dpr);
  ctx.scale(dpr, dpr);

  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  const pad = { top: 26, right: 24, bottom: 38, left: 54 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const maxY = Math.max(1.35, ...rows.map((r) => Math.max(r.output, r.wage, r.employment, r.automatedTasks * 2)));
  const minY = Math.min(0.55, ...rows.map((r) => Math.min(r.employment, r.wage)));

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = colors.text;
  ctx.font = "12px Inter, sans-serif";

  for (let i = 0; i <= 5; i += 1) {
    const y = pad.top + (chartH * i) / 5;
    const value = maxY - ((maxY - minY) * i) / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(2), 10, y + 4);
  }

  const xFor = (year) => pad.left + (year / params.horizon) * chartW;
  const yFor = (value) => pad.top + (1 - (value - minY) / (maxY - minY)) * chartH;
  drawSeries(ctx, rows, (r) => r.employment, xFor, yFor, colors.employment);
  drawSeries(ctx, rows, (r) => r.output, xFor, yFor, colors.output);
  drawSeries(ctx, rows, (r) => r.wage, xFor, yFor, colors.wage);
  drawSeries(ctx, rows, (r) => r.automatedTasks * 2, xFor, yFor, colors.exposed);

  ctx.strokeStyle = "#8a93a1";
  ctx.beginPath();
  ctx.moveTo(pad.left, yFor(1));
  ctx.lineTo(width - pad.right, yFor(1));
  ctx.stroke();

  ctx.fillStyle = colors.text;
  ctx.fillText("0", pad.left - 3, height - 12);
  ctx.fillText(`${params.horizon} years`, width - pad.right - 48, height - 12);
}

function drawSeries(ctx, rows, pick, xFor, yFor, color, width = 3) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  rows.forEach((row, index) => {
    const x = xFor(row.year);
    const y = yFor(pick(row));
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderBars(sectorEnd) {
  els.bars.innerHTML = "";
  sectorEnd
    .slice()
    .sort((a, b) => b.automated - a.automated)
    .forEach((sector) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-label"><span>${sector.name}</span><strong>${formatPct(sector.automated, 0)}</strong></div>
        <div class="bar-track"><div class="bar-fill" style="width: ${clamp(sector.automated) * 100}%"></div></div>
        <div class="bar-label"><span>Employment</span><strong>${formatSignedPct(sector.change)}</strong></div>
      `;
      els.bars.appendChild(row);
    });
}

function update() {
  renderOutputs();
  const result = simulate();
  const rows = result.rows;
  const last = rows[rows.length - 1];
  const peakUnemployment = Math.max(...rows.map((row) => row.unemployment));
  const threshold = rows.find((row) => row.instability > 0.16 + params.policyCushion * 0.08);

  els.employmentChange.textContent = formatSignedPct(last.employment - 1);
  els.peakUnemployment.textContent = formatPct(peakUnemployment);
  els.outputGain.textContent = formatSignedPct(last.output - 1);
  els.thresholdYear.textContent = threshold ? `Year ${threshold.year}` : "Not crossed";

  drawChart(rows);
  renderBars(result.sectorEnd);
  window.latestSimulation = rows;
}

function setPreset(kind) {
  if (kind === "baseline") {
    params = { ...defaults };
  }
  if (kind === "fast") {
    params = { ...defaults, performanceSpeed: 0.38, networkEffect: 0.55, capitalBottleneck: 0.12, tippingSharpness: 0.1, demandElasticity: 0.7 };
  }
  if (kind === "friction") {
    params = { ...defaults, performanceSpeed: 0.28, friction: 0.78, policyCushion: 0.08, demandElasticity: 0.34, capitalBottleneck: 0.36 };
  }
  document.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    input.value = key === "performanceSpeed" ? Math.round(params[key] * 100) : params[key];
  });
  update();
}

function exportCsv() {
  const rows = window.latestSimulation || simulate().rows;
  const header = "year,capability,employment_index,output_index,wage_index,transition_unemployment,automated_task_share,instability\n";
  const body = rows
    .map((row) => [row.year, row.capability, row.employment, row.output, row.wage, row.unemployment, row.automatedTasks, row.instability].join(","))
    .join("\n");
  const blob = new Blob([header + body], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "ai-labour-market-simulation.csv";
  link.click();
  URL.revokeObjectURL(url);
}

document.querySelectorAll("[data-param]").forEach((input) => {
  input.addEventListener("input", (event) => {
    const key = event.target.dataset.param;
    const raw = Number(event.target.value);
    params[key] = key === "performanceSpeed" ? raw / 100 : raw;
    if (key === "horizon") params[key] = raw;
    update();
  });
});

els.table.addEventListener("input", (event) => {
  const input = event.target;
  const index = Number(input.dataset.index);
  const field = input.dataset.field;
  if (!Number.isFinite(index) || !field) return;
  sectors[index][field] = field === "name" ? input.value : Number(input.value);
  update();
});

els.table.addEventListener("click", (event) => {
  const index = Number(event.target.dataset.delete);
  if (!Number.isFinite(index) || sectors.length <= 1) return;
  sectors.splice(index, 1);
  renderTable();
  update();
});

document.getElementById("addSectorBtn").addEventListener("click", () => {
  sectors.push({ name: "New sector", jobs: 5, exposure: 0.4, substitution: 0.25, complementarity: 0.2, threshold: 0.45, demand: 0.6, wage: 0.5 });
  renderTable();
  update();
});

document.getElementById("baselineBtn").addEventListener("click", () => setPreset("baseline"));
document.getElementById("fastBtn").addEventListener("click", () => setPreset("fast"));
document.getElementById("frictionBtn").addEventListener("click", () => setPreset("friction"));
document.getElementById("exportBtn").addEventListener("click", exportCsv);
window.addEventListener("resize", update);

renderTable();
update();
