/* ═══════════════════════════════════════════════════════════════════
   MCARE · dashboard.js
   Monte Carlo GBM engine — fully client-side
   Mirrors the Python logic in simulation/gbm.py + risk/metrics.py
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ─── Chart instances ───────────────────────────────────────────────
let chartPaths   = null;
let chartBacktest = null;
let chartDist    = null;

// ─── Last simulation result (for re-render) ────────────────────────
let lastResult   = null;

/* ════════════════════════════════════════════════════════════════════
   1. MATH UTILITIES
   ════════════════════════════════════════════════════════════════════ */

/** Seeded PRNG — mulberry32 (fast, good enough for demo) */
function makeRng(seed) {
  let s = seed >>> 0;
  return function () {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform — uniform pair → standard normal pair */
function boxMuller(u1, u2) {
  const mag = Math.sqrt(-2 * Math.log(u1 + 1e-15));
  return [
    mag * Math.cos(2 * Math.PI * u2),
    mag * Math.sin(2 * Math.PI * u2),
  ];
}

/** Compute a percentile of a sorted (or unsorted) array */
function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx    = p / 100 * (sorted.length - 1);
  const lo     = Math.floor(idx);
  const frac   = idx - lo;
  return lo + 1 < sorted.length
    ? sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    : sorted[lo];
}

function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function std(arr, mu) {
  const m = mu !== undefined ? mu : mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function skewness(arr, mu, s) {
  const m = mu !== undefined ? mu : mean(arr);
  const σ = s  !== undefined ? s  : std(arr, m);
  if (σ === 0) return 0;
  return arr.reduce((acc, v) => acc + ((v - m) / σ) ** 3, 0) / arr.length;
}

function kurtosis(arr, mu, s) {
  const m = mu !== undefined ? mu : mean(arr);
  const σ = s  !== undefined ? s  : std(arr, m);
  if (σ === 0) return 0;
  return arr.reduce((acc, v) => acc + ((v - m) / σ) ** 4, 0) / arr.length;
}

/* ════════════════════════════════════════════════════════════════════
   2. GBM SIMULATION  (mirrors simulation/gbm.py)
   ════════════════════════════════════════════════════════════════════ */

/**
 * Simulate GBM price paths.
 * Returns { paths: Float64Array[numPaths][horizon], terminalReturns: Float64Array }
 */
function simulateGBM({ S0, mu, sigma, horizon, numPaths, seed }) {
  const rng = makeRng(seed);

  /* We store a subset of full paths for visualisation (max 80 paths
     drawn, but ALL numPaths for metric computation). */
  const visualPaths = Math.min(numPaths, 80);

  // Full paths matrix — shape [horizon][numPaths] flattened row-major
  // For memory efficiency we only store ALL terminalPrices + visualPaths full curves
  const terminalPrices = new Float64Array(numPaths);

  // Store full price path for up to visualPaths paths
  const priceCurves = Array.from({ length: visualPaths }, () => new Float64Array(horizon));

  /* GBM: S_t = S0 * exp( (μ - σ²/2)·t + σ·W_t )
     We simulate path by path to stay memory-friendly in JS */
  for (let p = 0; p < numPaths; p++) {
    let logPrice = Math.log(S0);
    const drift  = (mu - 0.5 * sigma * sigma); // per-step drift
    let cumW     = 0;

    for (let t = 0; t < horizon; t++) {
      const u1 = rng();
      const u2 = rng();
      const [z] = boxMuller(u1, u2);
      cumW += z;
      const price = S0 * Math.exp(drift * (t + 1) + sigma * cumW);
      if (p < visualPaths) priceCurves[p][t] = price;
      if (t === horizon - 1) terminalPrices[p] = price;
    }
  }

  // Terminal simple returns: (S_T - S0) / S0
  const terminalReturns = new Float64Array(numPaths);
  for (let p = 0; p < numPaths; p++) {
    terminalReturns[p] = (terminalPrices[p] - S0) / S0;
  }

  // Mean path (across all visual paths)
  const meanPath = new Float64Array(horizon);
  for (let t = 0; t < horizon; t++) {
    let s = 0;
    for (let p = 0; p < visualPaths; p++) s += priceCurves[p][t];
    meanPath[t] = s / visualPaths;
  }

  return { priceCurves, meanPath, terminalReturns, terminalPrices };
}

/* ════════════════════════════════════════════════════════════════════
   3. RISK METRICS  (mirrors risk/metrics.py)
   ════════════════════════════════════════════════════════════════════ */

function computeVaR(returns, confidenceLevel) {
  // VaR = −Percentile(R, (1−α)·100)
  const p = (1 - confidenceLevel) * 100;
  return -percentile(Array.from(returns), p);
}

function computeCVaR(returns, confidenceLevel) {
  const varThreshold = -computeVaR(returns, confidenceLevel);
  const tail = Array.from(returns).filter(r => r <= varThreshold);
  if (tail.length === 0) return computeVaR(returns, confidenceLevel);
  return -mean(tail);
}

/* ════════════════════════════════════════════════════════════════════
   4. BACKTESTING
   ════════════════════════════════════════════════════════════════════ */

function runBacktest(terminalReturns, varValue, confidenceLevel) {
  const n         = terminalReturns.length;
  const threshold = -varValue; // returns below this are breaches
  let   breaches  = 0;

  const breachFlags = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (terminalReturns[i] < threshold) {
      breaches++;
      breachFlags[i] = 1;
    }
  }

  const observedRate    = breaches / n;
  const theoreticalRate = 1 - confidenceLevel;
  const accuracy        = 1 - Math.abs(observedRate - theoreticalRate) / (theoreticalRate + 1e-10);
  // Signal based on relative rate deviation — not raw count (which scales with path count)
  const relDev = theoreticalRate > 0
    ? Math.abs(observedRate - theoreticalRate) / theoreticalRate
    : 0;
  let signal = 'GREEN';
  if (relDev > 0.20) signal = 'YELLOW'; // >20% relative deviation from expected rate
  if (relDev > 0.50) signal = 'RED';    // >50% relative deviation — model is miscalibrated

  return { breaches, observedRate, theoreticalRate, accuracy, signal, breachFlags };
}

/* ════════════════════════════════════════════════════════════════════
   5. HISTOGRAM BUILDER
   ════════════════════════════════════════════════════════════════════ */

function buildHistogram(returns, bins = 60) {
  const arr  = Array.from(returns);
  const minV = Math.min(...arr);
  const maxV = Math.max(...arr);
  const size = (maxV - minV) / bins;
  const counts  = new Array(bins).fill(0);
  const centers = new Array(bins);
  for (let b = 0; b < bins; b++) centers[b] = minV + (b + 0.5) * size;
  for (const r of arr) {
    let b = Math.floor((r - minV) / size);
    if (b >= bins) b = bins - 1;
    counts[b]++;
  }
  return { centers, counts, size, minV, maxV };
}

/* ════════════════════════════════════════════════════════════════════
   6. ANIMATED NUMBER COUNTER
   ════════════════════════════════════════════════════════════════════ */

function animateNumber(el, targetStr, duration = 800) {
  const target = parseFloat(targetStr);
  if (isNaN(target)) { el.textContent = targetStr; return; }
  const decimals = (targetStr.split('.')[1] || '').length;
  const start    = performance.now();
  const from     = parseFloat(el.dataset.prev || '0');
  el.dataset.prev = targetStr;

  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    // Ease out cubic
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (target - from) * ease).toFixed(decimals);
    if (t < 1) requestAnimationFrame(tick);
    else el.textContent = targetStr;
  }
  requestAnimationFrame(tick);
}

/* ════════════════════════════════════════════════════════════════════
   7. CHART HELPERS
   ════════════════════════════════════════════════════════════════════ */

const ACCENT  = '#00FF85';
const RED     = '#FF3B47';
const ORANGE  = '#FF8C00';
const CYAN    = '#00D4FF';
const GRID    = '#1e232b';
const TEXT2   = '#7a8592';

function baseChartOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 900, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#111419',
        borderColor: '#1e232b',
        borderWidth: 1,
        titleColor: '#c8d0db',
        bodyColor: '#7a8592',
        titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
        bodyFont:  { family: "'JetBrains Mono', monospace", size: 10 },
        padding: 10,
      },
    },
    scales: {
      x: {
        grid:  { color: GRID, lineWidth: 0.5 },
        ticks: {
          color: TEXT2,
          font: { family: "'JetBrains Mono', monospace", size: 10 },
          maxTicksLimit: 8,
        },
        title: {
          display: !!xLabel,
          text: xLabel || '',
          color: TEXT2,
          font: { family: "'JetBrains Mono', monospace", size: 10 },
        },
        border: { color: GRID },
      },
      y: {
        grid:  { color: GRID, lineWidth: 0.5 },
        ticks: {
          color: TEXT2,
          font: { family: "'JetBrains Mono', monospace", size: 10 },
          maxTicksLimit: 6,
        },
        title: {
          display: !!yLabel,
          text: yLabel || '',
          color: TEXT2,
          font: { family: "'JetBrains Mono', monospace", size: 10 },
        },
        border: { color: GRID },
      },
    },
  };
}

/* ─── Price-paths chart ─────────────────────────────────────────── */
function renderPathsChart({ priceCurves, meanPath, horizon, varPrice, S0 }) {
  const ctx = document.getElementById('chart-paths').getContext('2d');
  if (chartPaths) chartPaths.destroy();

  const labels = Array.from({ length: horizon }, (_, i) => `T+${i + 1}`);

  const datasets = [];

  // Individual paths (faint)
  for (let p = 0; p < priceCurves.length; p++) {
    datasets.push({
      data: Array.from(priceCurves[p]),
      borderColor: 'rgba(0, 212, 255, 0.12)',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.2,
    });
  }

  // Mean path
  datasets.push({
    data: Array.from(meanPath),
    borderColor: ACCENT,
    borderWidth: 2.5,
    pointRadius: 0,
    tension: 0.3,
    shadowBlur: 8,
  });

  // VaR price level (horizontal reference)
  if (varPrice !== null) {
    datasets.push({
      data: Array(horizon).fill(varPrice),
      borderColor: RED,
      borderWidth: 1.5,
      borderDash: [4, 4],
      pointRadius: 0,
      tension: 0,
    });
  }

  const opts = baseChartOptions('Days', 'Price ($)');
  opts.animation = {
    duration: 1200,
    easing: 'easeOutQuart',
    onProgress(anim) {
      // Fade in paths
    },
  };

  chartPaths = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: opts,
  });

  document.getElementById('paths-overlay').classList.add('hidden');
}

/* ─── Backtest bar chart ────────────────────────────────────────── */
function renderBacktestChart({ observedRate, theoreticalRate, confLabel }) {
  const ctx = document.getElementById('chart-backtest').getContext('2d');
  if (chartBacktest) chartBacktest.destroy();

  const opts = baseChartOptions('', 'Rate');
  opts.plugins.tooltip.callbacks = {
    label: c => ` ${(c.raw * 100).toFixed(2)}%`,
  };
  opts.scales.y.ticks.callback = v => `${(v * 100).toFixed(1)}%`;
  opts.scales.x.grid = { display: false };
  opts.animation = { duration: 800, easing: 'easeOutQuart' };

  chartBacktest = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Observed Breach Rate', `Theoretical (${confLabel})`],
      datasets: [{
        data: [observedRate, theoreticalRate],
        backgroundColor: [
          observedRate > theoreticalRate * 1.3 ? RED : ACCENT,
          'rgba(0,212,255,0.4)',
        ],
        borderColor: [
          observedRate > theoreticalRate * 1.3 ? RED : ACCENT,
          CYAN,
        ],
        borderWidth: 1,
        borderRadius: 0,
        barThickness: 60,
      }],
    },
    options: opts,
  });

  document.getElementById('backtest-overlay').classList.add('hidden');
}

/* ─── Distribution histogram ────────────────────────────────────── */
function renderDistChart({ centers, counts, varValue, cvarValue }) {
  const ctx = document.getElementById('chart-dist').getContext('2d');
  if (chartDist) chartDist.destroy();

  // Build bar colors — red tail left of -VaR threshold
  const varThreshold = -varValue;
  const barColors    = centers.map(c => c < varThreshold ? 'rgba(255,59,71,0.65)' : 'rgba(0,255,133,0.3)');
  const borderColors = centers.map(c => c < varThreshold ? RED : 'rgba(0,255,133,0.6)');

  // Use centers as numeric x-data (scatter-bar hybrid via linear x-scale)
  const barData = centers.map((c, i) => ({ x: c, y: counts[i] }));

  const binWidth = centers.length > 1 ? Math.abs(centers[1] - centers[0]) : 0.005;

  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 900, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#111419',
        borderColor: '#1e232b',
        borderWidth: 1,
        titleColor: '#c8d0db',
        bodyColor: '#7a8592',
        titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
        bodyFont:  { family: "'JetBrains Mono', monospace", size: 10 },
        padding: 10,
        callbacks: {
          title: items => `Return: ${(items[0].parsed.x * 100).toFixed(2)}%`,
          label: item => ` Frequency: ${item.parsed.y}`,
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        grid: { color: GRID, lineWidth: 0.5 },
        ticks: {
          color: TEXT2,
          font: { family: "'JetBrains Mono', monospace", size: 10 },
          maxTicksLimit: 8,
          callback: v => `${(v * 100).toFixed(0)}%`,
        },
        title: {
          display: true, text: 'Return',
          color: TEXT2, font: { family: "'JetBrains Mono', monospace", size: 10 },
        },
        border: { color: GRID },
      },
      y: {
        grid: { color: GRID, lineWidth: 0.5 },
        ticks: {
          color: TEXT2,
          font: { family: "'JetBrains Mono', monospace", size: 10 },
          maxTicksLimit: 6,
        },
        title: {
          display: true, text: 'Frequency',
          color: TEXT2, font: { family: "'JetBrains Mono', monospace", size: 10 },
        },
        border: { color: GRID },
      },
    },
  };

  chartDist = new Chart(ctx, {
    type: 'bar',
    data: {
      datasets: [{
        label: 'Frequency',
        data: barData,
        backgroundColor: barColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 0,
        barThickness: 'flex',
        categoryPercentage: 1.0,
        barPercentage: 1.0,
      }],
    },
    options: opts,
    plugins: [{
      id: 'thresholdLines',
      afterDraw(chart) {
        const { ctx: c, chartArea, scales } = chart;
        const drawLine = (val, color, label) => {
          const x = scales.x.getPixelForValue(val);
          if (x < chartArea.left || x > chartArea.right) return;
          c.save();
          c.beginPath();
          c.moveTo(x, chartArea.top);
          c.lineTo(x, chartArea.bottom);
          c.strokeStyle = color;
          c.lineWidth = 1.5;
          c.setLineDash([5, 4]);
          c.stroke();
          c.fillStyle = color;
          c.font = "700 10px 'JetBrains Mono', monospace";
          c.textAlign = 'center';
          c.fillText(label, x, chartArea.top + 14);
          c.restore();
        };
        drawLine(-varValue,  RED,    `VaR ${(varValue  * 100).toFixed(1)}%`);
        drawLine(-cvarValue, ORANGE, `CVaR ${(cvarValue * 100).toFixed(1)}%`);
      },
    }],
  });

  document.getElementById('dist-overlay').classList.add('hidden');
}

/* ════════════════════════════════════════════════════════════════════
   8. UPDATE METRICS UI
   ════════════════════════════════════════════════════════════════════ */

function updateMetrics(result, confidenceLevel, ticker) {
  const { terminalReturns, varValue, cvarValue } = result;
  const rets = Array.from(terminalReturns);
  const mu   = mean(rets);
  const s    = std(rets, mu);

  const fmt = (v, d = 2) => (v * 100).toFixed(d);
  const fmtN = (v, d = 3) => v.toFixed(d);

  // Primary cards
  animateNumber(document.getElementById('val-var'),  fmt(varValue),  800);
  animateNumber(document.getElementById('val-cvar'), fmt(cvarValue), 800);
  const retSign = mu >= 0 ? '+' : '';
  document.getElementById('val-ret').textContent = retSign + fmt(mu) + '%';
  document.getElementById('val-ret').style.color = mu >= 0 ? 'var(--accent)' : 'var(--red)';
  animateNumber(document.getElementById('val-vol'), fmt(s), 800);

  // Secondary
  const rfDaily = 0.0001; // ~2.5% annual
  const sharpe  = s > 0 ? ((mu - rfDaily) / s) : 0;
  document.getElementById('val-sharpe').textContent = fmtN(sharpe, 3);
  document.getElementById('val-skew').textContent   = fmtN(skewness(rets, mu, s), 3);
  document.getElementById('val-kurt').textContent   = fmtN(kurtosis(rets, mu, s), 3);
  document.getElementById('val-min').textContent    = fmt(Math.min(...rets)) + '%';
  document.getElementById('val-max').textContent    = fmt(Math.max(...rets)) + '%';
  const pctProfit = (rets.filter(r => r > 0).length / rets.length * 100).toFixed(1) + '%';
  document.getElementById('val-profit').textContent = pctProfit;

  // Conf label
  const confLabel = (confidenceLevel * 100).toFixed(0) + '% CI';
  document.getElementById('metrics-conf-tag').textContent = confLabel;
  document.getElementById('desc-conf').textContent        = confLabel.replace(' CI', '');
}

/* ════════════════════════════════════════════════════════════════════
   9. UPDATE BACKTEST UI
   ════════════════════════════════════════════════════════════════════ */

function updateBacktest(bt, confidenceLevel) {
  const fmtPct = r => (r * 100).toFixed(2) + '%';
  document.getElementById('bt-observed').textContent    = fmtPct(bt.observedRate);
  document.getElementById('bt-theoretical').textContent = fmtPct(bt.theoreticalRate);
  document.getElementById('bt-count').textContent       = bt.breaches;
  document.getElementById('bt-accuracy').textContent    = (Math.max(0, bt.accuracy) * 100).toFixed(1) + '%';

  const signalColors = { GREEN: '#00FF85', YELLOW: '#FFD600', RED: '#FF3B47' };
  const sig = document.getElementById('bt-signal');
  sig.textContent = bt.signal;
  sig.style.color = signalColors[bt.signal];

  // Timeline grid (first 100 paths)
  const grid = document.getElementById('timeline-grid');
  grid.innerHTML = '';
  const show = Math.min(bt.breachFlags.length, 100);
  for (let i = 0; i < show; i++) {
    const cell = document.createElement('div');
    cell.className = 'timeline-cell ' + (bt.breachFlags[i] ? 'breach' : 'safe');
    cell.title = bt.breachFlags[i] ? `Path ${i + 1}: BREACH` : `Path ${i + 1}: SAFE`;
    grid.appendChild(cell);
  }

  renderBacktestChart({
    observedRate: bt.observedRate,
    theoreticalRate: bt.theoreticalRate,
    confLabel: (confidenceLevel * 100).toFixed(0) + '%',
  });
}

/* ════════════════════════════════════════════════════════════════════
   10. MAIN RUN HANDLER
   ════════════════════════════════════════════════════════════════════ */

function getInputs() {
  return {
    ticker:   document.getElementById('input-ticker').value.trim().toUpperCase() || 'SIM',
    S0:       parseFloat(document.getElementById('input-price').value),
    mu:       parseFloat(document.getElementById('input-mu').value),
    sigma:    parseFloat(document.getElementById('input-sigma').value),
    confLevel: parseFloat(document.querySelector('.toggle-btn.active').dataset.val),
    numPaths: parseInt(document.getElementById('input-paths').value, 10),
    horizon:  parseInt(document.getElementById('input-horizon').value, 10),
    seed:     parseInt(document.getElementById('input-seed').value, 10),
  };
}

function runSimulation() {
  const inp = getInputs();

  // Validate
  if (!inp.S0 || inp.S0 <= 0 || !inp.sigma || inp.sigma <= 0 || inp.numPaths < 100 || inp.horizon < 1) {
    flashError('Invalid inputs — check S₀ > 0, σ > 0, paths ≥ 100.');
    return;
  }

  // UI: running state
  const btn = document.getElementById('btn-run');
  btn.classList.add('running');
  btn.querySelector('.run-label').textContent = 'SIMULATING…';
  document.getElementById('nav-status').querySelector('.indicator-text').textContent = 'RUNNING';
  document.getElementById('nav-seed').textContent = inp.seed;

  showProgress();

  // Defer to next frame so UI updates render
  setTimeout(() => {
    try {
      // ── Run GBM
      const simResult = simulateGBM({
        S0:       inp.S0,
        mu:       inp.mu,
        sigma:    inp.sigma,
        horizon:  inp.horizon,
        numPaths: inp.numPaths,
        seed:     inp.seed,
      });

      // ── Risk Metrics
      const varValue  = computeVaR(simResult.terminalReturns, inp.confLevel);
      const cvarValue = computeCVaR(simResult.terminalReturns, inp.confLevel);
      const varPrice  = inp.S0 * (1 - varValue); // price @ VaR loss

      // ── Backtesting
      const bt = runBacktest(simResult.terminalReturns, varValue, inp.confLevel);

      // ── Histogram
      const hist = buildHistogram(simResult.terminalReturns, 60);

      // Store
      lastResult = { simResult, varValue, cvarValue, varPrice, bt, hist, inp };

      // ── Render charts
      renderPathsChart({
        priceCurves: simResult.priceCurves,
        meanPath:    simResult.meanPath,
        horizon:     inp.horizon,
        varPrice,
        S0:          inp.S0,
      });

      renderDistChart({ ...hist, varValue, cvarValue });

      // ── Update metrics UI
      updateMetrics({ terminalReturns: simResult.terminalReturns, varValue, cvarValue }, inp.confLevel, inp.ticker);

      // ── Update backtest
      updateBacktest(bt, inp.confLevel);

      // ── Update paths tag
      document.getElementById('paths-tag').textContent = `${inp.numPaths.toLocaleString()} PATHS · T=${inp.horizon}d`;

      // ── Done
      hideProgress();
      btn.classList.remove('running');
      btn.querySelector('.run-label').textContent = 'RUN SIMULATION';
      document.getElementById('nav-status').querySelector('.indicator-text').textContent = 'LIVE';

      // Scroll to metrics
      document.getElementById('section-metrics').scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
      hideProgress();
      btn.classList.remove('running');
      btn.querySelector('.run-label').textContent = 'RUN SIMULATION';
      flashError('Simulation error: ' + err.message);
    }
  }, 60);
}

/* ─── Progress animation ──────────────────────────────────────────  */
let progressTimer = null;
function showProgress() {
  const wrap = document.getElementById('progress-wrap');
  const bar  = document.getElementById('progress-bar');
  const lbl  = document.getElementById('progress-label');
  wrap.style.display = 'block';
  bar.style.width = '0%';
  let pct = 0;
  const msgs = ['Seeding RNG…', 'Generating Brownian paths…', 'Computing GBM…', 'Calculating metrics…', 'Building charts…'];
  let msgIdx = 0;
  progressTimer = setInterval(() => {
    pct = Math.min(pct + Math.random() * 18, 90);
    bar.style.width = pct + '%';
    if (pct > (msgIdx + 1) * 18) {
      lbl.textContent = msgs[Math.min(msgIdx++, msgs.length - 1)];
    }
  }, 120);
}
function hideProgress() {
  if (progressTimer) clearInterval(progressTimer);
  const bar  = document.getElementById('progress-bar');
  const wrap = document.getElementById('progress-wrap');
  bar.style.width = '100%';
  setTimeout(() => { wrap.style.display = 'none'; }, 400);
}

function flashError(msg) {
  console.error(msg);
  const btn = document.getElementById('btn-run');
  btn.classList.remove('running');
  btn.style.background = 'var(--red)';
  btn.querySelector('.run-label').textContent = msg;
  setTimeout(() => {
    btn.style.background = '';
    btn.querySelector('.run-label').textContent = 'RUN SIMULATION';
  }, 3000);
}

/* ════════════════════════════════════════════════════════════════════
   11. SCROLL REVEAL  (IntersectionObserver)
   ════════════════════════════════════════════════════════════════════ */

function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

/* ════════════════════════════════════════════════════════════════════
   12. BOOT
   ════════════════════════════════════════════════════════════════════ */

function initConfidenceToggle() {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

function initPathsLabel() {
  const pathsInput = document.getElementById('input-paths');
  const pathsLabel = document.getElementById('run-paths-label');
  pathsInput.addEventListener('input', () => {
    const v = parseInt(pathsInput.value, 10);
    pathsLabel.textContent = isNaN(v) ? '? PATHS' : `${v.toLocaleString()} PATHS`;
  });
}

function initRunButton() {
  document.getElementById('btn-run').addEventListener('click', runSimulation);
  // Run on Enter within any input
  document.querySelectorAll('.field-input').forEach(el => {
    el.addEventListener('keydown', e => { if (e.key === 'Enter') runSimulation(); });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  initConfidenceToggle();
  initPathsLabel();
  initRunButton();

  // Auto-run a demo on load so charts aren't empty
  setTimeout(runSimulation, 600);
});
