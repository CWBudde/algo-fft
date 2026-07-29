// app.js — wiring for the algofft Signal Lab.
//
// Talks to globalThis.algofft (namespaced bridge exported from main.go) and
// draws through window.Render (render.js). Every call into the wasm module
// is guarded on the Go side and returns {error, panic} on failure — this
// file must never let that throw past the render loop.
(() => {
  "use strict";

  // ---------------------------------------------------------------------
  // DOM refs — see the DOM CONTRACT comment at the top of index.html.
  // ---------------------------------------------------------------------

  const rack = document.getElementById("rack");
  const statusEl = document.getElementById("status");
  const phaseWheelEl = document.getElementById("phaseWheel");

  const waveCanvas = document.getElementById("wave");
  const spectrumCanvas = document.getElementById("spectrum");
  const gridCanvas = document.getElementById("grid");
  const reconstructionCanvas = document.getElementById("reconstruction");
  const convolutionCanvas = document.getElementById("convolution");

  const sizeSelect = document.getElementById("size");
  const customNInput = document.getElementById("customN");
  const nPresets = document.getElementById("nPresets");
  const gridSizeSelect = document.getElementById("gridSize");
  const freqAInput = document.getElementById("freqA");
  const freqBInput = document.getElementById("freqB");
  const noiseInput = document.getElementById("noise");
  const animateInput = document.getElementById("animate");
  const playButton = document.getElementById("play");
  const randomizeButton = document.getElementById("randomize");
  const signalTypeSelect = document.getElementById("signalType");
  const windowSelect = document.getElementById("window");
  const precisionSelect = document.getElementById("precision");
  const strategySelect = document.getElementById("strategy");
  const convKernelSelect = document.getElementById("convKernel");
  const magScaleSelect = document.getElementById("magScale");
  const freqScaleSelect = document.getElementById("freqScale");

  const telemetryN = document.getElementById("telemetryN");
  const telemetryWindow = document.getElementById("telemetryWindow");
  const telemetryAlgorithm = document.getElementById("telemetryAlgorithm");
  const telemetryTime = document.getElementById("telemetryTime");
  const telemetryRoundtrip = document.getElementById("telemetryRoundtrip");
  const reconError = document.getElementById("reconError");

  const valueByKey = {};
  document.querySelectorAll("[data-value]").forEach((span) => {
    valueByKey[span.dataset.value] = span;
  });

  const reducedMotion = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
    : false;

  // ---------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------

  function updateValue(key, value) {
    if (!valueByKey[key]) return;
    valueByKey[key].textContent =
      typeof value === "number" ? value.toString() : value;
  }

  function setStatus(text, state) {
    // state: "loading" | "ready" | "error"
    statusEl.textContent = text;
    statusEl.dataset.state = state || "loading";
  }

  function formatNs(ns) {
    if (!Number.isFinite(ns)) return "—";
    if (ns < 1000) return `${ns.toFixed(0)} ns`;
    if (ns < 1e6) return `${(ns / 1000).toFixed(2)} µs`;
    return `${(ns / 1e6).toFixed(2)} ms`;
  }

  function formatError(v) {
    if (!Number.isFinite(v)) return "—";
    return v.toExponential(1);
  }

  // precisionValue translates the <select> value ("64"/"128") to the exact
  // string the Go bridge's precisionKind.String() produces. Passing the raw
  // "64"/"128" through is unsafe: plancache.go's precisionFromString treats
  // "64" as a complex128 request (a real bug in the current bridge), so this
  // demo always sends the canonical "complex64"/"complex128" names instead.
  function precisionValue(selectValue) {
    // Handles both the static HTML options ("64"/"128", before info()
    // repopulates the <select> with the bridge's canonical names) and the
    // canonical "complex64"/"complex128" strings used afterward.
    return selectValue === "128" || selectValue === "complex128"
      ? "complex128"
      : "complex64";
  }

  // ---------------------------------------------------------------------
  // API call wrapper — every call into wasm degrades to a status message
  // rather than throwing.
  // ---------------------------------------------------------------------

  // call invokes a namespaced algofft export and never throws. Failures for
  // core calls (analyze/info) are surfaced in #status per the demo contract;
  // failures for secondary panels (silent: true) are only logged, so one
  // not-yet-landed export (e.g. convolve, while WP4/WP5 are still in
  // flight) does not permanently steal the global status line from the
  // parts of the page that are working.
  function call(name, opts, callOpts) {
    const silent = callOpts && callOpts.silent;
    const api = window.algofft;
    if (!api || typeof api[name] !== "function") {
      if (silent) console.warn(`algofft.${name} is not available`);
      else setStatus(`algofft.${name} is not available`, "error");
      return null;
    }
    let result;
    try {
      result = api[name](opts);
    } catch (err) {
      console.error(err);
      if (silent) console.warn(`${name} threw`, err);
      else setStatus(`${name} threw: ${err && err.message ? err.message : err}`, "error");
      return null;
    }
    if (result && result.error) {
      if (silent) console.warn(`${name}: ${result.error}`);
      else setStatus(result.error, "error");
      return null;
    }
    return result;
  }

  // ---------------------------------------------------------------------
  // Reusable output buffers (WP0 zero-copy contract): {f32, u8} pairs over
  // one JS-owned ArrayBuffer, reused across frames and reallocated only
  // when a length changes.
  // ---------------------------------------------------------------------

  const sinks = {}; // key -> {f32, u8, len}

  function makeSink(len) {
    const buf = new ArrayBuffer(Math.max(1, len) * 4);
    return { f32: new Float32Array(buf), u8: new Uint8Array(buf) };
  }

  function sink(key, len) {
    const existing = sinks[key];
    if (existing && existing.len === len) return existing;
    const fresh = makeSink(len);
    fresh.len = len;
    sinks[key] = fresh;
    return fresh;
  }

  function outBuffers(n, magCount, gridCount) {
    return {
      signal: sink("signal", n),
      spectrum: sink("spectrum", magCount),
      phase: sink("phase", magCount),
      window: sink("window", n),
      reconstruction: sink("reconstruction", n),
      gridSpectrum: sink("gridSpectrum", Math.max(1, gridCount)),
      gridPhase: sink("gridPhase", Math.max(1, gridCount)),
    };
  }

  // ---------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------

  const state = {
    n: Number(sizeSelect.value) || 1024,
    gridSize: Number(gridSizeSelect.value) || 64,
    freqA: Number(freqAInput.value),
    freqB: Number(freqBInput.value),
    noise: Number(noiseInput.value),
    phase: 0,
    animate: animateInput.checked,
    playing: false,
    signalType: signalTypeSelect.value,
    windowName: windowSelect.value,
    precision: precisionValue(precisionSelect.value),
    strategy: strategySelect.value,
    convKernel: convKernelSelect.value,
    magScale: magScaleSelect ? magScaleSelect.value : "db",
    freqScale: freqScaleSelect ? freqScaleSelect.value : "linear",
  };

  let info = null;
  let lastMainSig = null;
  let lastGridSig = null;
  let lastAudioSig = null;
  let lastMainResult = null;
  let lastGridData = null; // {mags, phases, size}
  let lastConvKey = null;
  let wasmReady = false;

  let audioCtx = null;
  let audioSource = null;
  let audioGain = null;

  // ---------------------------------------------------------------------
  // Strategy / precision option population from algofft.info()
  // ---------------------------------------------------------------------

  function humanize(name) {
    if (!name) return name;
    return name === "auto" ? "Auto" : name;
  }

  function populateStrategySelect(strategies) {
    if (!strategies || !strategies.length) return;
    const previous = state.strategy;
    strategySelect.innerHTML = "";
    strategies.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = humanize(name);
      strategySelect.appendChild(opt);
    });
    const match = strategies.find((s) => s === previous) || strategies[0];
    strategySelect.value = match;
    state.strategy = match;
  }

  function populatePrecisionSelect(precisions) {
    if (!precisions || !precisions.length) return;
    const previousShort = precisionSelect.value; // "64" | "128"
    precisionSelect.innerHTML = "";
    precisions.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name; // canonical: "complex64" / "complex128"
      opt.textContent = name;
      precisionSelect.appendChild(opt);
    });
    const wanted = precisionValue(previousShort);
    precisionSelect.value = precisions.includes(wanted) ? wanted : precisions[0];
    state.precision = precisionSelect.value;
  }

  // ---------------------------------------------------------------------
  // Signatures — decide what actually needs recomputation.
  // ---------------------------------------------------------------------

  function mainSignature() {
    return JSON.stringify([
      state.n,
      state.precision,
      state.strategy,
      state.signalType,
      state.windowName,
      state.freqA,
      state.freqB,
      state.noise,
    ]);
  }

  function gridSignature() {
    return JSON.stringify([
      state.gridSize,
      state.precision,
      state.strategy,
      state.freqA,
      state.freqB,
      state.noise,
    ]);
  }

  function audioSignature() {
    // Deliberately excludes phase: audio must not rebuild every animation
    // frame just because the visual phase is sweeping. See updateAudio().
    return JSON.stringify([
      state.n,
      state.precision,
      state.strategy,
      state.signalType,
      state.windowName,
      state.freqA,
      state.freqB,
      state.noise,
    ]);
  }

  function convSignature() {
    return JSON.stringify([state.convKernel, state.precision, state.n]);
  }

  // ---------------------------------------------------------------------
  // Compute + draw
  // ---------------------------------------------------------------------

  function baseOpts(extra) {
    return Object.assign(
      {
        n: state.n,
        precision: state.precision,
        strategy: state.strategy,
        signal: state.signalType,
        window: state.windowName,
        freqA: state.freqA,
        freqB: state.freqB,
        noise: state.noise,
        phase: state.phase,
      },
      extra
    );
  }

  function recomputeIfNeeded() {
    if (!wasmReady) return;

    const sig1D = mainSignature();
    const sigGrid = gridSignature();
    const needMain = state.animate || sig1D !== lastMainSig;
    const needGrid = state.animate || sigGrid !== lastGridSig;

    if (!needMain && !needGrid) return;

    const magCount = Math.max(1, Math.floor(state.n / 2));
    const gridCount = state.gridSize * state.gridSize;
    const out = outBuffers(state.n, magCount, gridCount);

    const opts = baseOpts({
      gridSize: state.gridSize,
      grid: needGrid,
      roundtrip: needMain,
      out,
    });

    const result = call("analyze", opts);
    if (!result) return;

    lastMainSig = sig1D;
    if (needGrid) lastGridSig = sigGrid;

    lastMainResult = result;
    if (needGrid && result.gridSpectrum) {
      lastGridData = {
        mags: result.gridSpectrum,
        phases: result.gridPhase || null,
        size: result.gridSize || state.gridSize,
      };
    }

    setStatus("WASM ready", "ready");
    redrawMain();
    if (needGrid) redrawGrid();
    updateTelemetry(result);

    if (state.playing) updateAudio();
  }

  function redrawMain() {
    if (!lastMainResult) return;
    const r = lastMainResult;
    window.Render.drawWave(waveCanvas, r.signal, r.window || null);
    window.Render.drawSpectrum(spectrumCanvas, r.spectrum, r.phase, {
      magScale: state.magScale,
      freqScale: state.freqScale,
    });
    window.Render.drawReconstruction(
      reconstructionCanvas,
      r.signal,
      r.reconstruction || null
    );
  }

  function redrawGrid() {
    if (!lastGridData) return;
    window.Render.drawGrid(
      gridCanvas,
      lastGridData.mags,
      lastGridData.phases,
      lastGridData.size
    );
  }

  function updateTelemetry(result) {
    telemetryN.textContent = String(result.n);

    const windowLabel = result.windowName || state.windowName;
    telemetryWindow.textContent = windowLabel;

    const plan = result.plan || {};
    const algorithm = plan.algorithm || "—";
    const requested = plan.strategyRequested;
    const resolved = plan.strategy;
    let algoText = algorithm;
    if (
      requested &&
      resolved &&
      requested.toLowerCase() !== "auto" &&
      requested.toLowerCase() !== resolved.toLowerCase()
    ) {
      algoText = `${algorithm} (requested ${requested})`;
    }
    telemetryAlgorithm.textContent = algoText;

    // A single transform at these sizes is usually faster than the browser's
    // clamped clock can resolve, so the raw reading is often exactly 0. Report
    // that as an upper bound rather than printing "0 ns", which reads as a bug
    // rather than as the measurement limit it actually is.
    const timing = result.timing || {};
    const granularity = info ? info.timerGranularityNs : 0;
    let timeText;
    if (
      Number.isFinite(timing.forwardNs) &&
      granularity > 0 &&
      timing.forwardNs < granularity
    ) {
      timeText = `< ${formatNs(granularity)} (timer limit)`;
    } else {
      timeText = formatNs(timing.forwardNs);
    }
    telemetryTime.textContent = timeText;

    const rt = result.roundtrip;
    if (rt && Number.isFinite(rt.maxAbsError)) {
      telemetryRoundtrip.textContent = formatError(rt.maxAbsError);
      reconError.textContent = formatError(rt.maxAbsError);
    } else {
      telemetryRoundtrip.textContent = "—";
      reconError.textContent = "—";
    }
  }

  // ---------------------------------------------------------------------
  // Convolution panel — recomputed only when its own parameters change.
  // ---------------------------------------------------------------------

  function recomputeConvIfNeeded() {
    if (!wasmReady) return;
    const key = convSignature();
    if (key === lastConvKey) return;

    const result = call(
      "convolve",
      {
        n: state.n,
        precision: state.precision,
        kernel: state.convKernel,
        signal: state.signalType,
        freqA: state.freqA,
        freqB: state.freqB,
        noise: state.noise,
      },
      { silent: true }
    );
    lastConvKey = key; // don't hammer a missing/failing export every frame
    if (!result || !result.result) return;

    window.Render.drawConvolution(
      convolutionCanvas,
      result.result,
      result.lagZeroIndex
    );
  }

  // ---------------------------------------------------------------------
  // Audio — rebuild the buffer source only when signal parameters actually
  // change, never on every animation frame (that clicks audibly).
  // ---------------------------------------------------------------------

  function updateAudio() {
    if (!audioCtx || !state.playing) return;

    const sig = audioSignature();
    if (sig === lastAudioSig && audioSource) return; // nothing to rebuild

    // A dedicated, phase-frozen render for audio: the visual phase sweep
    // must not tear the loop down every frame.
    const magCount = Math.max(1, Math.floor(state.n / 2));
    const out = outBuffers(state.n, magCount, 1);
    const result = call(
      "analyze",
      baseOpts({ phase: 0, grid: false, roundtrip: false, out })
    );
    if (!result || !result.signal) return;

    lastAudioSig = sig;

    if (audioSource) {
      audioSource.stop();
      audioSource.disconnect();
      audioSource = null;
    }

    const signal = result.signal;
    const buffer = audioCtx.createBuffer(1, signal.length, audioCtx.sampleRate);
    const channel = buffer.getChannelData(0);
    let peak = 1e-9;
    for (let i = 0; i < signal.length; i++) peak = Math.max(peak, Math.abs(signal[i]));
    const norm = 0.3 / peak;
    for (let i = 0; i < signal.length; i++) channel[i] = signal[i] * norm;

    audioSource = audioCtx.createBufferSource();
    audioSource.buffer = buffer;
    audioSource.loop = true;
    audioSource.connect(audioGain);
    audioSource.start();
  }

  function toggleAudio() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioGain = audioCtx.createGain();
      audioGain.gain.value = 0.8;
      audioGain.connect(audioCtx.destination);
    }

    if (state.playing) {
      state.playing = false;
      playButton.textContent = "Play loop";
      if (audioSource) {
        audioSource.stop();
        audioSource.disconnect();
        audioSource = null;
      }
      lastAudioSig = null;
      return;
    }

    state.playing = true;
    playButton.textContent = "Stop audio";
    updateAudio();
  }

  // ---------------------------------------------------------------------
  // Control wiring
  // ---------------------------------------------------------------------

  function syncNFromCustom() {
    const v = Number(customNInput.value);
    const lo = (info && info.minN) || 2;
    const hi = (info && info.maxN) || 1048576;
    if (Number.isFinite(v) && v >= lo && v <= hi) {
      state.n = Math.round(v);
    } else {
      state.n = Number(sizeSelect.value) || state.n;
    }
    updatePresetChips();
  }

  function updatePresetChips() {
    nPresets.querySelectorAll(".chip").forEach((chip) => {
      const n = Number(chip.dataset.n);
      chip.setAttribute("aria-pressed", n === state.n ? "true" : "false");
    });
  }

  function readControls() {
    state.gridSize = Number(gridSizeSelect.value);
    state.freqA = Number(freqAInput.value);
    state.freqB = Number(freqBInput.value);
    state.noise = Number(noiseInput.value);
    state.animate = animateInput.checked;
    state.signalType = signalTypeSelect.value;
    state.windowName = windowSelect.value;
    state.precision = precisionValue(precisionSelect.value);
    state.strategy = strategySelect.value;
    state.convKernel = convKernelSelect.value;

    updateValue("freqA", state.freqA);
    updateValue("freqB", state.freqB);
    updateValue("noise", state.noise.toFixed(2));

    syncNFromCustom();
    recomputeIfNeeded();
    recomputeConvIfNeeded();
  }

  function randomize() {
    const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    state.freqA = rand(2, 24);
    state.freqB = rand(18, 96);
    state.noise = Math.round((Math.random() * 0.25 + 0.02) * 100) / 100;

    freqAInput.value = state.freqA;
    freqBInput.value = state.freqB;
    noiseInput.value = state.noise;
  }

  function wireControls() {
    sizeSelect.addEventListener("change", () => {
      customNInput.value = sizeSelect.value;
      readControls();
    });
    customNInput.addEventListener("input", readControls);
    gridSizeSelect.addEventListener("change", readControls);
    freqAInput.addEventListener("input", readControls);
    freqBInput.addEventListener("input", readControls);
    noiseInput.addEventListener("input", readControls);
    animateInput.addEventListener("change", readControls);
    signalTypeSelect.addEventListener("change", readControls);
    windowSelect.addEventListener("change", readControls);
    precisionSelect.addEventListener("change", readControls);
    strategySelect.addEventListener("change", readControls);
    convKernelSelect.addEventListener("change", readControls);

    // Scale changes are presentation only — no transform needs re-running,
    // so redraw from the cached result instead of calling into wasm.
    if (magScaleSelect) {
      magScaleSelect.addEventListener("change", () => {
        state.magScale = magScaleSelect.value;
        redrawMain();
      });
    }
    if (freqScaleSelect) {
      freqScaleSelect.addEventListener("change", () => {
        state.freqScale = freqScaleSelect.value;
        redrawMain();
      });
    }

    playButton.addEventListener("click", toggleAudio);
    randomizeButton.addEventListener("click", () => {
      randomize();
      readControls();
    });

    nPresets.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        customNInput.value = chip.dataset.n;
        sizeSelect.value = chip.dataset.n;
        readControls();
      });
    });
  }

  // ---------------------------------------------------------------------
  // Resize handling — redraw cached data, never recompute the transform
  // just because the canvas box changed.
  // ---------------------------------------------------------------------

  function wireResize() {
    const canvases = [
      waveCanvas,
      spectrumCanvas,
      gridCanvas,
      reconstructionCanvas,
      convolutionCanvas,
    ];

    let pending = false;
    function scheduleRedraw() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        redrawMain();
        redrawGrid();
      });
    }

    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(scheduleRedraw);
      canvases.forEach((c) => ro.observe(c));
    } else {
      window.addEventListener("resize", scheduleRedraw);
    }

    window.Render.watchDPR(() => {
      window.Render.invalidateRamp();
      scheduleRedraw();
    });
  }

  // ---------------------------------------------------------------------
  // Animation loop — only touches wasm when something needs it.
  // ---------------------------------------------------------------------

  function tick() {
    if (state.animate) {
      state.phase += 0.03;
    }
    recomputeIfNeeded();
    requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------------
  // Boot sequence
  // ---------------------------------------------------------------------

  // ensureRenderLoaded guarantees window.Render exists before it is used.
  // index.html is out of scope for this file to edit, so render.js is
  // loaded defensively here rather than assumed to already be wired in via
  // a <script> tag.
  function ensureRenderLoaded() {
    if (window.Render) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "render.js";
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("failed to load render.js"));
      document.head.appendChild(s);
    });
  }

  async function loadWasmWithProgress(onProgress) {
    if (!WebAssembly.instantiateStreaming) {
      WebAssembly.instantiateStreaming = async (resp, importObject) => {
        const source = await (await resp).arrayBuffer();
        return WebAssembly.instantiate(source, importObject);
      };
    }

    const go = new Go();
    const response = await fetch("algofft.wasm");

    if (!response.body || !response.body.getReader || reducedMotion) {
      onProgress(1);
      const result = await WebAssembly.instantiateStreaming(response, go.importObject);
      return { go, result };
    }

    const total = Number(response.headers.get("content-length")) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total > 0) onProgress(Math.min(0.98, received / total));
    }
    onProgress(1);

    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }

    const result = await WebAssembly.instantiate(bytes, go.importObject);
    return { go, result };
  }

  function boot() {
    populateStrategySelect(info && info.strategies);
    populatePrecisionSelect(info && info.precisions);

    customNInput.value = String(state.n);
    updatePresetChips();

    wireControls();
    wireResize();

    readControls();
    requestAnimationFrame(tick);

    rack.dataset.boot = "ready";
  }

  async function initWasm() {
    setStatus("Initializing WASM…", "loading");

    await ensureRenderLoaded();

    const { go, result } = await loadWasmWithProgress((p) => {
      window.Render.phaseWheel(phaseWheelEl, p);
    });

    go.run(result.instance); // main() ends in select{} and never resolves

    // Give the Go side one tick to register globalThis.algofft.
    await new Promise((resolve) => setTimeout(resolve, 0));

    wasmReady = true;
    phaseWheelEl.dataset.state = "ready";

    info = call("info", undefined);
    setStatus("WASM ready", "ready");

    boot();
  }

  initWasm().catch((err) => {
    console.error(err);
    setStatus(
      "WebAssembly failed to load. Check that algofft.wasm is served as application/wasm.",
      "error"
    );
  });
})();
