/*
 * bench.js — benchmark page controller.
 *
 * This page owns no wasm instance of its own. Everything Go-side runs in
 * bench-worker.js, which means the main thread stays free to paint the chart,
 * respond to Stop and keep the table interactive while a sweep is running, and
 * — more importantly for the numbers — nothing the UI does lands inside a
 * timed region.
 *
 * Two behaviours are worth reading the code for rather than guessing at:
 *
 *   Stop. The worker's benchmark loop yields between bounded Go calls, so a
 *   posted "cancel" is dispatched at the next gap and the run ends with its
 *   partial results intact. worker.terminate() plus a respawn is kept behind a
 *   watchdog for the case where a single Go call runs long enough that the gap
 *   never comes.
 *
 *   runId. Every run carries a monotonic id and every worker message is
 *   checked against it. A re-run started while a stale worker is still
 *   draining its queue must not append rows to the new run's table.
 */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const statusEl = $("status");
  const runButton = $("run-bench");
  const stopButton = $("stop-bench");
  const resultsBody = $("results-body");
  const resultsHead = $("results-head");
  const estimateEl = $("bench-estimate");
  const progressEl = $("bench-progress");
  const progressFill = $("bench-progress-fill");
  const caseCountEl = $("case-count");
  const doneCountEl = $("done-count");
  const granularityEl = $("timerGranularity");
  const liveRegion = $("live-region");
  const exportCsvButton = $("export-csv");
  const exportJsonButton = $("export-json");

  const SIZE_OPTIONS = [
    { n: 64 },
    { n: 128 },
    { n: 256 },
    { n: 512 },
    { n: 1000, badge: "mixed" },
    { n: 1009, badge: "rader" },
    { n: 1024 },
    { n: 2048 },
    { n: 4096 },
    { n: 8192 },
    { n: 16384 },
    { n: 32768 },
    { n: 65536 },
    { n: 131072 },
  ];

  const EFFORT = {
    quick: {
      label: "Quick",
      trials: 3,
      targetMs: 20,
      sizes: [64, 256, 1024, 4096, 16384],
    },
    standard: {
      label: "Standard",
      trials: 5,
      targetMs: 50,
      sizes: [64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    },
    thorough: {
      label: "Thorough",
      trials: 7,
      targetMs: 120,
      sizes: [
        64, 128, 256, 512, 1000, 1009, 1024, 2048, 4096, 8192, 16384, 32768,
        65536,
      ],
    },
  };

  /** Refuse anything larger outright; the Go side caps at 256 as a backstop. */
  const MAX_CASES = 120;
  /** Ask before starting a run estimated to take longer than this. */
  const CONFIRM_SECONDS = 120;

  const BOOT_TIMEOUT_MS = 30000;
  const CANCEL_TIMEOUT_MS = 1500;
  const LIVE_THROTTLE_MS = 700;

  let worker = null;
  let workerReady = false;
  let info = null;
  let bootTimer = null;
  let cancelTimer = null;
  let bootAttempts = 0;

  let runId = 0;
  let running = false;
  let expectedTotal = 0;
  let completedCases = 0;

  /** Result rows for the current (or last) run. */
  let rows = [];
  /** Cross-run state carried into a row by the "prepared" message. */
  const preparedByIndex = new Map();

  let sortKey = "size";
  let sortDir = 1;
  let lastLive = 0;

  let chart = null;

  const selection = {
    effort: "standard",
    sizes: new Set(EFFORT.standard.sizes),
    precisions: new Set(["complex64"]),
    strategies: new Set(["auto"]),
    planner: "estimate",
  };

  /* ------------------------------------------------------------------ *
   * formatting
   * ------------------------------------------------------------------ */

  const fmt = () => globalThis.BenchChart;

  function formatNs(ns) {
    return fmt().formatTimeNs(ns);
  }

  /**
   * Plan-construction times are single measurements, not batches, so they land
   * on multiples of the timer clamp — and a fast build reads as a literal zero.
   * Saying "below one clock tick" is the honest rendering; printing "0 ns"
   * would claim a precision the browser does not offer.
   */
  function formatPlanNs(ns) {
    if (!(ns > 0)) return "< 1 tick";
    return formatNs(ns);
  }

  function formatSeconds(s) {
    if (s < 10) return `${s.toFixed(1)} s`;
    if (s < 90) return `${s.toFixed(0)} s`;
    return `${(s / 60).toFixed(1)} min`;
  }

  function setStatus(text, state) {
    statusEl.textContent = text;
    statusEl.dataset.state = state;
  }

  function announce(text, force) {
    const now = Date.now();
    if (!force && now - lastLive < LIVE_THROTTLE_MS) return;
    lastLive = now;
    liveRegion.textContent = text;
  }

  /* ------------------------------------------------------------------ *
   * chips
   * ------------------------------------------------------------------ */

  function makeChip(label, pressed, badge) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.setAttribute("aria-pressed", String(pressed));
    button.textContent = label;
    if (badge) button.dataset.badge = badge;
    return button;
  }

  function bindToggle(button, set, value, atLeastOne) {
    button.addEventListener("click", () => {
      if (set.has(value)) {
        if (atLeastOne && set.size === 1) return;
        set.delete(value);
      } else {
        set.add(value);
      }
      button.setAttribute("aria-pressed", String(set.has(value)));
      refreshEstimate();
    });
  }

  function buildSizeChips() {
    const host = $("size-chips");
    host.textContent = "";
    for (const option of SIZE_OPTIONS) {
      const chip = makeChip(
        String(option.n),
        selection.sizes.has(option.n),
        option.badge,
      );
      bindToggle(chip, selection.sizes, option.n, true);
      chip.dataset.size = String(option.n);
      host.appendChild(chip);
    }
  }

  function syncSizeChips() {
    document.querySelectorAll("#size-chips .chip").forEach((chip) => {
      chip.setAttribute(
        "aria-pressed",
        String(selection.sizes.has(Number(chip.dataset.size))),
      );
    });
  }

  function buildPrecisionChips(precisions) {
    const host = $("precision-chips");
    host.textContent = "";
    for (const precision of precisions) {
      const chip = makeChip(precision, selection.precisions.has(precision));
      bindToggle(chip, selection.precisions, precision, true);
      host.appendChild(chip);
    }
  }

  function buildStrategyChips(strategies) {
    const host = $("strategy-chips");
    host.textContent = "";
    for (const strategy of strategies) {
      const chip = makeChip(strategy, selection.strategies.has(strategy));
      bindToggle(chip, selection.strategies, strategy, true);
      host.appendChild(chip);
    }
  }

  function buildPlannerSelect(modes) {
    const select = $("planner-select");
    select.textContent = "";
    for (const mode of modes) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = mode;
      select.appendChild(option);
    }
    select.value = selection.planner;
    select.addEventListener("change", () => {
      selection.planner = select.value;
      refreshEstimate();
    });
  }

  function buildEffortChips() {
    document.querySelectorAll("#effort-chips .chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        selection.effort = chip.dataset.effort;
        document
          .querySelectorAll("#effort-chips .chip")
          .forEach((other) =>
            other.setAttribute("aria-pressed", String(other === chip)),
          );
        // Mutated in place, never replaced: the chip handlers close over this
        // Set, and swapping it for a fresh one leaves every chip toggling an
        // orphan.
        selection.sizes.clear();
        for (const n of EFFORT[selection.effort].sizes) selection.sizes.add(n);
        syncSizeChips();
        refreshEstimate();
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * request assembly and estimation
   * ------------------------------------------------------------------ */

  function buildRequest() {
    const effort = EFFORT[selection.effort];
    return {
      sizes: [...selection.sizes].sort((a, b) => a - b),
      precisions: [...selection.precisions],
      strategies: [...selection.strategies],
      planner: selection.planner,
      trials: effort.trials,
      targetMs: effort.targetMs,
    };
  }

  function caseCount(request) {
    return (
      request.sizes.length * request.precisions.length * request.strategies.length
    );
  }

  /**
   * Rough wall-clock estimate, deliberately on the pessimistic side.
   *
   * Per case: calibration converges from below and costs about two target
   * windows, then `trials` windows are timed. Plan construction is charged
   * twice (the cold build plus the wisdom-warm rebuild) at a crude per-element
   * rate. The floor on the target window is the timer clamp — the Go side
   * raises a too-small target to 200 × the probed granularity, and the
   * estimate has to know that or it will understate every Quick run on
   * Firefox.
   */
  function estimateSeconds(request) {
    const granularityMs = info ? info.timerGranularityNs / 1e6 : 0.1;
    const target = Math.max(request.targetMs, 200 * granularityMs);
    let total = 0;
    for (const n of request.sizes) {
      const planMs = 2 * Math.max(0.3, n * 4e-4);
      const perCase = (request.trials + 2) * target + planMs;
      total += perCase * request.precisions.length * request.strategies.length;
    }
    return total / 1000;
  }

  function refreshEstimate() {
    const request = buildRequest();
    const count = caseCount(request);
    const seconds = estimateSeconds(request);

    caseCountEl.textContent = String(count);

    const effort = EFFORT[selection.effort];
    $("effort-detail").textContent =
      `${effort.trials} trials × ${effort.targetMs} ms target`;

    if (count === 0) {
      estimateEl.textContent = "nothing selected";
      runButton.disabled = true;
      return;
    }

    if (count > MAX_CASES) {
      estimateEl.textContent = `${count} cases — over the ${MAX_CASES} case cap`;
      runButton.disabled = true;
      return;
    }

    estimateEl.textContent = `${count} cases · est. ${formatSeconds(seconds)}`;
    runButton.disabled = running || !workerReady;
  }

  /* ------------------------------------------------------------------ *
   * worker lifecycle
   * ------------------------------------------------------------------ */

  function spawnWorker() {
    if (worker) {
      worker.terminate();
      worker = null;
    }

    workerReady = false;
    bootAttempts += 1;
    setStatus("Starting worker…", "loading");

    try {
      worker = new Worker("bench-worker.js");
    } catch (err) {
      setStatus(`Worker failed to start: ${err.message}`, "error");
      return;
    }

    worker.onmessage = onWorkerMessage;
    worker.onerror = (event) => {
      setStatus(
        `Worker error: ${event.message || "see console"} — the page must be served over HTTP, not opened from a file:// URL.`,
        "error",
      );
      workerReady = false;
      runButton.disabled = true;
    };

    clearTimeout(bootTimer);
    bootTimer = setTimeout(() => {
      if (workerReady) return;
      if (bootAttempts < 2) {
        spawnWorker();
        return;
      }
      setStatus("WASM failed to load in the worker", "error");
    }, BOOT_TIMEOUT_MS);
  }

  function onWorkerMessage(event) {
    const message = event.data || {};

    switch (message.type) {
      case "ready":
        clearTimeout(bootTimer);
        workerReady = true;
        onReady(message.info);
        break;

      case "jobStarted":
        if (message.runId !== runId) return;
        expectedTotal = message.total;
        caseCountEl.textContent = String(message.total);
        granularityEl.textContent = `${(message.granularityNs / 1000).toFixed(1)} µs`;
        break;

      case "jobPrepared":
        if (message.runId !== runId) return;
        preparedByIndex.set(message.caseIndex, message.prepared);
        break;

      case "jobResult":
        if (message.runId !== runId) return;
        addResult(message.result, message.caseIndex);
        break;

      case "jobProgress":
        if (message.runId !== runId) return;
        updateProgress(message);
        break;

      case "jobError":
        if (message.runId !== runId) return;
        setStatus(`Benchmark error: ${message.error}`, "error");
        break;

      case "runDone":
        if (message.runId !== runId) return;
        finishRun(message.cancelled);
        break;

      case "fatal":
        setStatus(`Worker fatal: ${message.error}`, "error");
        workerReady = false;
        finishRun(true);
        break;

      default:
        break;
    }
  }

  function onReady(workerInfo) {
    info = workerInfo || {};

    granularityEl.textContent = `${(info.timerGranularityNs / 1000).toFixed(1)} µs`;
    $("method-granularity").textContent =
      `${(info.timerGranularityNs / 1000).toFixed(1)} µs (${info.timerGranularityNs.toFixed(0)} ns)`;
    $("method-simd").textContent = info.simd ? "SIMD enabled" : "no SIMD";

    buildPrecisionChips(info.precisions || ["complex64", "complex128"]);
    buildStrategyChips(info.strategies || ["auto"]);
    buildPlannerSelect(info.plannerModes || ["estimate"]);

    setStatus(`WASM ready · ${info.version || "devel"}`, "ready");
    refreshEstimate();
  }

  /* ------------------------------------------------------------------ *
   * run control
   * ------------------------------------------------------------------ */

  function startRun() {
    if (!workerReady || running) return;

    const request = buildRequest();
    const count = caseCount(request);
    if (count === 0 || count > MAX_CASES) return;

    const seconds = estimateSeconds(request);
    if (seconds > CONFIRM_SECONDS) {
      const ok = window.confirm(
        `This sweep is ${count} cases and should take about ${formatSeconds(seconds)}.\n\nRun it?`,
      );
      if (!ok) return;
    }

    runId += 1;
    running = true;
    rows = [];
    preparedByIndex.clear();
    completedCases = 0;
    expectedTotal = count;

    renderTable();
    chart.setRows(rows);

    runButton.disabled = true;
    stopButton.disabled = false;
    exportCsvButton.disabled = true;
    exportJsonButton.disabled = true;
    doneCountEl.textContent = "0";
    setProgress(0);
    setStatus("Running…", "running");
    announce(`Benchmark started, ${count} cases`, true);

    $("method-trials").textContent = String(request.trials);

    worker.postMessage({ type: "run", runId, request });
  }

  function requestStop() {
    if (!running) return;
    stopButton.disabled = true;
    setStatus("Stopping…", "running");
    worker.postMessage({ type: "cancel", runId });

    // If the worker is stuck inside a single very long Go call it will never
    // reach the yield where the cancel is dispatched. Terminating loses the
    // wasm instance (and any wisdom it accumulated) but keeps every row that
    // already made it across, which is the trade the user asked for by
    // pressing Stop.
    clearTimeout(cancelTimer);
    cancelTimer = setTimeout(() => {
      if (!running) return;
      setStatus("Worker did not stop — restarting it", "error");
      spawnWorker();
      finishRun(true);
    }, CANCEL_TIMEOUT_MS);
  }

  function finishRun(cancelled) {
    clearTimeout(cancelTimer);
    if (!running) return;
    running = false;
    stopButton.disabled = true;
    runButton.disabled = !workerReady;
    exportCsvButton.disabled = rows.length === 0;
    exportJsonButton.disabled = rows.length === 0;

    if (cancelled) {
      setStatus(
        `Stopped · ${rows.length} of ${expectedTotal} cases kept`,
        "ready",
      );
      announce(`Benchmark stopped with ${rows.length} results kept`, true);
    } else {
      setProgress(1);
      setStatus(`Done · ${rows.length} cases`, "ready");
      announce(`Benchmark finished, ${rows.length} results`, true);
    }
  }

  function setProgress(fraction) {
    const pct = Math.max(0, Math.min(1, fraction)) * 100;
    progressFill.style.width = `${pct}%`;
    progressEl.setAttribute("aria-valuenow", pct.toFixed(0));
  }

  function updateProgress(message) {
    const total = message.total || expectedTotal || 1;
    const trials = message.trials || 5;
    // Each case is one prepare step, a handful of calibration steps and
    // `trials` timed steps. Calibration length is not known in advance, so it
    // is charged a flat slice; the bar is monotone within a case either way.
    let within = 0;
    if (message.phase === "calibrate") within = 0.2;
    else if (message.phase === "trial") within = 0.25 + 0.75 * (message.trial / trials);
    else if (message.phase === "finished") within = 1;

    setProgress((message.caseIndex + within) / total);

    if (message.phase === "trial") {
      announce(
        `Case ${message.caseIndex + 1} of ${total}, trial ${message.trial} of ${trials}`,
      );
    }
  }

  function addResult(result, caseIndex) {
    const prepared = preparedByIndex.get(caseIndex);
    if (prepared && !result.algorithm) {
      result.algorithm = prepared.algorithm;
      result.strategyResolved = prepared.strategyResolved;
    }
    rows.push(result);
    completedCases += 1;
    doneCountEl.textContent = String(completedCases);
    renderTable();
    chart.setRows(rows);
  }

  /* ------------------------------------------------------------------ *
   * table
   * ------------------------------------------------------------------ */

  function derived(row, key) {
    switch (key) {
      case "throughput":
        return fmt().throughputMBs(row);
      case "normalized":
        return fmt().normalizedNs(row);
      case "strategy":
        return `${row.strategyRequested} → ${row.strategyResolved || ""}`;
      case "reliable":
        return row.reliable ? 1 : 0;
      default:
        return row[key];
    }
  }

  function sortedRows() {
    const numeric = resultsHead.querySelector(`th[data-key="${sortKey}"]`)
      ?.dataset.numeric === "true";
    return rows.slice().sort((a, b) => {
      const av = derived(a, sortKey);
      const bv = derived(b, sortKey);
      if (numeric) return (Number(av) - Number(bv)) * sortDir;
      return String(av).localeCompare(String(bv)) * sortDir;
    });
  }

  function cell(text, options) {
    const td = document.createElement("td");
    td.textContent = text;
    if (options && options.warn) td.className = "warn";
    if (options && options.title) td.title = options.title;
    return td;
  }

  function renderTable() {
    resultsBody.textContent = "";

    if (!rows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 12;
      td.style.textAlign = "center";
      td.style.color = "var(--dim)";
      td.textContent = running
        ? "Measuring…"
        : "Run the benchmark to populate this table.";
      tr.appendChild(td);
      resultsBody.appendChild(tr);
      return;
    }

    for (const row of sortedRows()) {
      const tr = document.createElement("tr");

      if (row.error) {
        tr.appendChild(cell(String(row.size)));
        tr.appendChild(cell(row.precision || ""));
        tr.appendChild(cell(row.strategyRequested || ""));
        const td = cell(row.error, { warn: true });
        td.colSpan = 9;
        tr.appendChild(td);
        resultsBody.appendChild(tr);
        continue;
      }

      const downgraded =
        row.strategyResolved && row.strategyResolved !== row.strategyRequested;

      tr.appendChild(cell(String(row.size)));
      tr.appendChild(cell(row.precision));
      tr.appendChild(
        cell(`${row.strategyRequested} → ${row.strategyResolved || "?"}`, {
          warn: downgraded,
          title: downgraded
            ? "The planner did not honour the requested strategy at this size."
            : "",
        }),
      );
      tr.appendChild(cell(row.algorithm || "?"));
      tr.appendChild(cell(formatNs(row.medianNs), { warn: !row.reliable }));
      tr.appendChild(cell(`${(100 * (row.relStddev || 0)).toFixed(1)}%`));
      tr.appendChild(cell(row.iterations.toLocaleString("en-US")));
      tr.appendChild(cell(fmt().throughputMBs(row).toFixed(1)));
      tr.appendChild(cell(fmt().normalizedNs(row).toFixed(3)));
      tr.appendChild(cell(formatPlanNs(row.planNs)));
      tr.appendChild(cell(formatPlanNs(row.planWisdomNs)));
      tr.appendChild(cell(row.reliable ? "yes" : "no", { warn: !row.reliable }));

      resultsBody.appendChild(tr);
    }
  }

  function bindSorting() {
    resultsHead.querySelectorAll("th").forEach((th) => {
      const button = th.querySelector("button");
      if (!button) return;
      button.addEventListener("click", () => {
        const key = th.dataset.key;
        if (sortKey === key) sortDir = -sortDir;
        else {
          sortKey = key;
          sortDir = 1;
        }
        resultsHead
          .querySelectorAll("th")
          .forEach((other) => other.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
        renderTable();
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * export
   * ------------------------------------------------------------------ */

  const CSV_COLUMNS = [
    "size",
    "precision",
    "strategyRequested",
    "strategyResolved",
    "algorithm",
    "planner",
    "medianNs",
    "meanNs",
    "stddevNs",
    "relStddev",
    "iterations",
    "totalNs",
    "granularityNs",
    "reliable",
    "planNs",
    "planWisdomNs",
    "checksum",
  ];

  function csvEscape(value) {
    const text = value == null ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function download(filename, mime, text) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function exportCsv() {
    const header = [...CSV_COLUMNS, "throughputMBs", "nsPerNLogN"].join(",");
    const lines = sortedRows().map((row) =>
      [
        ...CSV_COLUMNS.map((key) => csvEscape(row[key])),
        fmt().throughputMBs(row).toFixed(4),
        fmt().normalizedNs(row).toFixed(6),
      ].join(","),
    );
    download("algofft-bench.csv", "text/csv", [header, ...lines].join("\n"));
  }

  function exportJson() {
    const payload = {
      generated: new Date().toISOString(),
      userAgent: navigator.userAgent,
      info,
      request: buildRequest(),
      note:
        "WebAssembly: no SIMD, no threads. Times are medians of whole batches; " +
        "plan construction is excluded from the transform time.",
      results: sortedRows(),
    };
    download(
      "algofft-bench.json",
      "application/json",
      JSON.stringify(payload, null, 2),
    );
  }

  /* ------------------------------------------------------------------ *
   * boot
   * ------------------------------------------------------------------ */

  chart = globalThis.BenchChart.create({
    canvas: $("bench-chart"),
    legend: $("bench-legend"),
    tooltip: $("bench-tooltip"),
    modeButtons: document.querySelectorAll("#bench-modes button"),
  });
  chart.setRows([]);

  buildSizeChips();
  buildEffortChips();
  bindSorting();
  refreshEstimate();

  runButton.addEventListener("click", startRun);
  stopButton.addEventListener("click", requestStop);
  exportCsvButton.addEventListener("click", exportCsv);
  exportJsonButton.addEventListener("click", exportJson);

  window.addEventListener("beforeunload", () => {
    if (worker) worker.terminate();
  });

  spawnWorker();
})();
