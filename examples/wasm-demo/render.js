// render.js — canvas rendering for the algofft Signal Lab.
//
// Signature idea: phase is hue. This file owns the phase -> hue ramp as the
// single source of colour truth: it reads the four cyclic anchors
// (--phase-0 --phase-90 --phase-180 --phase-270) from computed CSS custom
// properties rather than hardcoding colours, so the ramp always matches
// style.css.
//
// No modules — everything hangs off window.Render.
(() => {
  "use strict";

  // ---------------------------------------------------------------------
  // Phase -> hue ramp
  // ---------------------------------------------------------------------

  let rampCache = null;

  function parseColor(str) {
    // Accepts "#rrggbb" or "rgb(r,g,b)"; getComputedStyle normalizes most
    // browsers to rgb(...) but the CSS source is #hex, so support both.
    str = str.trim();
    if (str.startsWith("#")) {
      const hex = str.slice(1);
      const full =
        hex.length === 3
          ? hex
              .split("")
              .map((c) => c + c)
              .join("")
          : hex;
      const n = parseInt(full, 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    const m = str.match(/rgba?\(([^)]+)\)/);
    if (m) {
      const parts = m[1].split(",").map((s) => parseFloat(s));
      return [parts[0] || 0, parts[1] || 0, parts[2] || 0];
    }
    return [255, 0, 255]; // loud fallback so a parse failure is obvious
  }

  // readRamp (re)reads the four cyclic anchors from :root. Cached, but call
  // invalidateRamp() if the theme could have changed underneath us.
  function readRamp() {
    if (rampCache) return rampCache;
    const cs = getComputedStyle(document.documentElement);
    const anchors = [
      parseColor(cs.getPropertyValue("--phase-0") || "#ff5fa2"),
      parseColor(cs.getPropertyValue("--phase-90") || "#ffc24b"),
      parseColor(cs.getPropertyValue("--phase-180") || "#39e5c8"),
      parseColor(cs.getPropertyValue("--phase-270") || "#6c6bff"),
    ];
    rampCache = anchors;
    return anchors;
  }

  function invalidateRamp() {
    rampCache = null;
  }

  // rampColor maps a phase in radians (-pi, pi] (or any real value, taken
  // mod 2pi) to an [r,g,b] triple by interpolating cyclically through the
  // four anchors, wrapping 270deg -> 0deg (360deg).
  function rampColorRGB(phaseRad) {
    const anchors = readRamp();
    let t = phaseRad / (2 * Math.PI); // turns
    t = t - Math.floor(t); // wrap to [0,1)
    const seg = t * 4; // which quarter, 0..4
    const i = Math.min(3, Math.floor(seg));
    const frac = seg - i;
    const a = anchors[i];
    const b = anchors[(i + 1) % 4];
    return [
      Math.round(a[0] + (b[0] - a[0]) * frac),
      Math.round(a[1] + (b[1] - a[1]) * frac),
      Math.round(a[2] + (b[2] - a[2]) * frac),
    ];
  }

  // rampColor returns a CSS rgb()/rgba() string for a phase in radians.
  // Optional lightness in [0,1] scales toward black (domain colouring).
  function rampColor(phaseRad, lightness) {
    const [r, g, b] = rampColorRGB(phaseRad);
    if (lightness === undefined) return `rgb(${r}, ${g}, ${b})`;
    const l = Math.max(0, Math.min(1, lightness));
    return `rgb(${Math.round(r * l)}, ${Math.round(g * l)}, ${Math.round(b * l)})`;
  }

  // ---------------------------------------------------------------------
  // DPR-correct canvas setup
  // ---------------------------------------------------------------------

  // Tracks the last DPR used per canvas so callers can cheaply detect
  // whether a re-fit is needed.
  const dprState = new WeakMap();

  function currentDPR() {
    return window.devicePixelRatio || 1;
  }

  // fitCanvas sizes canvas.width/height to the element's CSS box scaled by
  // devicePixelRatio, and sets a single transform so all subsequent drawing
  // happens in CSS pixels. Call once per resize/DPR-change, not per frame.
  // Returns {width, height} in CSS pixels for convenience.
  function fitCanvas(canvas) {
    const dpr = currentDPR();
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const pxW = Math.max(1, Math.round(w * dpr));
    const pxH = Math.max(1, Math.round(h * dpr));

    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    dprState.set(canvas, dpr);

    return { width: w, height: h, ctx };
  }

  // ensureFit re-fits a canvas only when its CSS size or the DPR changed.
  // Cheap to call every frame.
  function ensureFit(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = currentDPR();
    const wantW = Math.max(1, Math.round(rect.width * dpr));
    const wantH = Math.max(1, Math.round(rect.height * dpr));
    if (
      canvas.width !== wantW ||
      canvas.height !== wantH ||
      dprState.get(canvas) !== dpr
    ) {
      return fitCanvas(canvas);
    }
    const ctx = canvas.getContext("2d");
    // Transform is reset whenever width/height is *assigned*, but we did not
    // touch it here, so nothing to redo. Still return ctx + CSS size.
    return { width: Math.round(rect.width), height: Math.round(rect.height), ctx };
  }

  // Re-read devicePixelRatio on change, not once at load (matchMedia trick).
  function watchDPR(onChange) {
    let mql = null;
    function arm() {
      const dpr = currentDPR();
      if (mql) mql.removeEventListener?.("change", handleChange);
      mql = matchMedia(`(resolution: ${dpr}dppx)`);
      mql.addEventListener?.("change", handleChange);
    }
    function handleChange() {
      onChange(currentDPR());
      arm();
    }
    arm();
  }

  // ---------------------------------------------------------------------
  // Small drawing helpers
  // ---------------------------------------------------------------------

  function readVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fallback;
  }

  function clear(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
  }

  // compactMag keeps a magnitude tick label inside the axis gutter. Peak
  // magnitudes grow with n, so plain toFixed(2) overflows and gets clipped.
  function compactMag(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(1);
    return v.toPrecision(2);
  }

  function drawAxisFrame(ctx, w, h, pad) {
    ctx.strokeStyle = readVar("--rule", "#262c4e");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, h - pad.b);
    ctx.lineTo(w - pad.r, h - pad.b);
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, h - pad.b);
    ctx.stroke();
  }

  function tickLabel(ctx, text, x, y, align) {
    ctx.save();
    ctx.fillStyle = readVar("--dim", "#8189b8");
    ctx.font =
      "10px " +
      (readVar("--font-mono", "monospace") || "monospace");
    ctx.textAlign = align || "center";
    ctx.textBaseline = "top";
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  // ---------------------------------------------------------------------
  // drawSpectrum — peak-hold aggregation to pixel columns, phase-coloured.
  // ---------------------------------------------------------------------

  // opts: {
  //   magScale: "linear" | "db",
  //   freqScale: "linear" | "log",
  //   sampleRate: number (nominal, defaults to n so axis reads in bins),
  // }
  function drawSpectrum(canvas, mags, phases, opts) {
    opts = opts || {};
    const { width: w, height: h, ctx } = ensureFit(canvas);
    clear(ctx, w, h);
    if (!mags || mags.length === 0) return;

    // The left gutter has to fit the widest magnitude tick label. Magnitudes
    // scale with n, so a fixed 36px clipped the leading digit ("514.47" drew
    // as "14.47"). Labels are formatted compactly below and the gutter is
    // sized to match.
    const pad = { l: 52, r: 8, t: 10, b: 20 };
    const plotW = Math.max(1, w - pad.l - pad.r);
    const plotH = Math.max(1, h - pad.t - pad.b);
    const nBins = mags.length;
    const magScale = opts.magScale === "db" ? "db" : "linear";
    const freqScale = opts.freqScale === "log" ? "log" : "linear";

    // Precompute magnitude transform.
    let maxMag = 1e-9;
    for (let i = 0; i < nBins; i++) if (mags[i] > maxMag) maxMag = mags[i];
    const maxDb = 20 * Math.log10(maxMag + 1e-12);
    const floorDb = maxDb - 90; // 90 dB dynamic range

    function magToNorm(m) {
      if (magScale === "db") {
        const db = 20 * Math.log10(m + 1e-12);
        return Math.max(0, Math.min(1, (db - floorDb) / (maxDb - floorDb || 1)));
      }
      return Math.max(0, Math.min(1, m / maxMag));
    }

    // Map a bin index to an x pixel column, honouring freq scale.
    function binToX(i) {
      if (freqScale === "log") {
        const lo = Math.log10(1);
        const hiv = Math.log10(nBins);
        const v = Math.log10(Math.max(1, i));
        return pad.l + ((v - lo) / (hiv - lo || 1)) * plotW;
      }
      return pad.l + (i / (nBins - 1 || 1)) * plotW;
    }

    // Peak-hold aggregation: for each pixel column, find the bin(s) that map
    // there and keep the one with the largest magnitude. We walk bins once
    // and bucket by rounded pixel column.
    const colPeakMag = new Float32Array(plotW + 1).fill(-1);
    const colPeakPhase = new Float32Array(plotW + 1);
    for (let i = 0; i < nBins; i++) {
      const x = binToX(i);
      const col = Math.max(0, Math.min(plotW, Math.round(x - pad.l)));
      if (mags[i] > colPeakMag[col]) {
        colPeakMag[col] = mags[i];
        colPeakPhase[col] = phases ? phases[i] : 0;
      }
    }

    for (let col = 0; col <= plotW; col++) {
      if (colPeakMag[col] < 0) continue;
      const norm = magToNorm(colPeakMag[col]);
      const barH = norm * plotH;
      const x = pad.l + col;
      const y = pad.t + (plotH - barH);
      ctx.fillStyle = rampColor(colPeakPhase[col]);
      ctx.fillRect(x, y, 1, barH);
    }

    drawAxisFrame(ctx, w, h, pad);

    // Axis ticks: frequency (bin index) on x, magnitude on y.
    const xTicks = 5;
    for (let t = 0; t <= xTicks; t++) {
      const frac = t / xTicks;
      let binIdx;
      if (freqScale === "log") {
        const lo = 0;
        const hiv = Math.log10(nBins);
        binIdx = Math.round(Math.pow(10, lo + frac * (hiv - lo)));
      } else {
        binIdx = Math.round(frac * (nBins - 1));
      }
      const x = pad.l + frac * plotW;
      ctx.strokeStyle = readVar("--rule", "#262c4e");
      ctx.beginPath();
      ctx.moveTo(x, h - pad.b);
      ctx.lineTo(x, h - pad.b + 3);
      ctx.stroke();
      tickLabel(ctx, String(binIdx), x, h - pad.b + 5, "center");
    }

    const yTicks = 3;
    for (let t = 0; t <= yTicks; t++) {
      const frac = t / yTicks;
      const y = pad.t + (1 - frac) * plotH;
      ctx.strokeStyle = readVar("--rule", "#262c4e");
      ctx.beginPath();
      ctx.moveTo(pad.l - 3, y);
      ctx.lineTo(pad.l, y);
      ctx.stroke();
      const label =
        magScale === "db"
          ? `${Math.round(floorDb + frac * (maxDb - floorDb))}`
          : compactMag(frac * maxMag);
      ctx.save();
      ctx.fillStyle = readVar("--dim", "#8189b8");
      ctx.font = "10px " + (readVar("--font-mono", "monospace") || "monospace");
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(label, pad.l - 5, y);
      ctx.restore();
    }
  }

  // ---------------------------------------------------------------------
  // drawWave — waveform plus a faint window envelope overlay.
  // ---------------------------------------------------------------------

  function drawWave(canvas, signal, windowShape) {
    const { width: w, height: h, ctx } = ensureFit(canvas);
    clear(ctx, w, h);
    if (!signal || signal.length === 0) return;

    const pad = { l: 8, r: 8, t: 8, b: 8 };
    const plotW = Math.max(1, w - pad.l - pad.r);
    const plotH = Math.max(1, h - pad.t - pad.b);
    const midY = pad.t + plotH / 2;
    const n = signal.length;

    let maxAbs = 1e-9;
    for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(signal[i]));
    maxAbs = Math.max(maxAbs, 1e-9);

    // Faint window envelope, both polarities, drawn first (behind the trace).
    if (windowShape && windowShape.length === n) {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / (n - 1 || 1)) * plotW;
        const y = midY - windowShape[i] * (plotH * 0.45);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      for (let i = n - 1; i >= 0; i--) {
        const x = pad.l + (i / (n - 1 || 1)) * plotW;
        const y = midY + windowShape[i] * (plotH * 0.45);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(129, 137, 184, 0.12)"; // --dim, faint
      ctx.fill();
    }

    // Zero line.
    ctx.strokeStyle = readVar("--rule", "#262c4e");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, midY);
    ctx.lineTo(pad.l + plotW, midY);
    ctx.stroke();

    // Waveform trace.
    ctx.strokeStyle = rampColor(Math.PI / 2); // amber-ish anchor, stable colour
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.l + (i / (n - 1 || 1)) * plotW;
      const y = midY - (signal[i] / maxAbs) * (plotH * 0.45);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ---------------------------------------------------------------------
  // drawReconstruction — original + reconstructed + error curve.
  // ---------------------------------------------------------------------

  function drawReconstruction(canvas, original, reconstructed) {
    const { width: w, height: h, ctx } = ensureFit(canvas);
    clear(ctx, w, h);
    if (!original || original.length === 0) return;

    const pad = { l: 8, r: 8, t: 8, b: 8 };
    const plotW = Math.max(1, w - pad.l - pad.r);
    const plotH = Math.max(1, h - pad.t - pad.b);
    const midY = pad.t + plotH / 2;
    const n = original.length;
    const hasRecon = reconstructed && reconstructed.length === n;

    let maxAbs = 1e-9;
    for (let i = 0; i < n; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(original[i]));
      if (hasRecon) maxAbs = Math.max(maxAbs, Math.abs(reconstructed[i]));
    }

    ctx.strokeStyle = readVar("--rule", "#262c4e");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, midY);
    ctx.lineTo(pad.l + plotW, midY);
    ctx.stroke();

    function trace(data, color, width) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / (n - 1 || 1)) * plotW;
        const y = midY - (data[i] / maxAbs) * (plotH * 0.45);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Original: dim reference.
    trace(original, "rgba(232, 236, 255, 0.35)", 1.5); // --lume, faint

    if (hasRecon) {
      trace(reconstructed, rampColor(0), 1.25); // magenta anchor

      // Error curve, scaled up so it is visible, drawn at the bottom third.
      let maxErr = 1e-9;
      const err = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        err[i] = reconstructed[i] - original[i];
        maxErr = Math.max(maxErr, Math.abs(err[i]));
      }
      const errBandY = pad.t + plotH * 0.85;
      const errBandH = plotH * 0.12;
      ctx.strokeStyle = rampColor(Math.PI); // cyan anchor
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = pad.l + (i / (n - 1 || 1)) * plotW;
        const y = errBandY - (err[i] / maxErr) * (errBandH / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  // ---------------------------------------------------------------------
  // drawGrid — proper domain colouring: hue = phase, lightness = log|mag|.
  // Renders to a square viewport; never stretches non-square canvases.
  // ---------------------------------------------------------------------

  const gridOffscreens = new WeakMap();

  function drawGrid(canvas, mags, phases, size) {
    const { width: w, height: h, ctx } = ensureFit(canvas);
    clear(ctx, w, h);
    if (!mags || !size) return;

    let off = gridOffscreens.get(canvas);
    if (!off || off.size !== size) {
      const c = document.createElement("canvas");
      c.width = size;
      c.height = size;
      off = { canvas: c, ctx: c.getContext("2d"), size };
      gridOffscreens.set(canvas, off);
    }

    const hasPhase = phases && phases.length === mags.length;
    const logMags = new Float32Array(mags.length);
    let maxLog = -Infinity;
    for (let i = 0; i < mags.length; i++) {
      const v = Math.log10(mags[i] + 1e-6);
      logMags[i] = v;
      if (v > maxLog) maxLog = v;
    }
    if (!Number.isFinite(maxLog) || maxLog <= -6) maxLog = 1;
    // Use a fixed-width dynamic range below the peak so quiet bins don't all
    // crush to black.
    const minLog = maxLog - 6;

    const image = off.ctx.createImageData(size, size);
    const half = Math.floor(size / 2);
    for (let y = 0; y < size; y++) {
      const srcY = (y + half) % size;
      for (let x = 0; x < size; x++) {
        const srcX = (x + half) % size;
        const srcIndex = srcY * size + srcX;
        const lightness = Math.max(
          0,
          Math.min(1, (logMags[srcIndex] - minLog) / (maxLog - minLog || 1))
        );
        const phase = hasPhase ? phases[srcIndex] : 0;
        const [r, g, b] = rampColorRGB(phase);
        const idx = (y * size + x) * 4;
        image.data[idx] = Math.round(r * lightness);
        image.data[idx + 1] = Math.round(g * lightness);
        image.data[idx + 2] = Math.round(b * lightness);
        image.data[idx + 3] = 255;
      }
    }
    off.ctx.putImageData(image, 0, 0);

    // Square viewport: fit the largest centered square within the CSS box,
    // never stretch.
    const side = Math.min(w, h);
    const ox = (w - side) / 2;
    const oy = (h - side) / 2;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off.canvas, ox, oy, side, side);

    ctx.strokeStyle = readVar("--rule", "#262c4e");
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
  }

  // ---------------------------------------------------------------------
  // drawConvolution — convolution/correlation result trace with a lag-zero
  // marker.
  // ---------------------------------------------------------------------

  function drawConvolution(canvas, result, lagZeroIndex) {
    const { width: w, height: h, ctx } = ensureFit(canvas);
    clear(ctx, w, h);
    if (!result || result.length === 0) return;

    const pad = { l: 8, r: 8, t: 8, b: 8 };
    const plotW = Math.max(1, w - pad.l - pad.r);
    const plotH = Math.max(1, h - pad.t - pad.b);
    const midY = pad.t + plotH / 2;
    const n = result.length;

    let maxAbs = 1e-9;
    for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(result[i]));

    ctx.strokeStyle = readVar("--rule", "#262c4e");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, midY);
    ctx.lineTo(pad.l + plotW, midY);
    ctx.stroke();

    ctx.strokeStyle = rampColor(Math.PI / 2);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.l + (i / (n - 1 || 1)) * plotW;
      const y = midY - (result[i] / maxAbs) * (plotH * 0.45);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (typeof lagZeroIndex === "number" && lagZeroIndex >= 0 && lagZeroIndex < n) {
      const x = pad.l + (lagZeroIndex / (n - 1 || 1)) * plotW;
      ctx.strokeStyle = rampColor(Math.PI);
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      tickLabel(ctx, "lag 0", x, pad.t + 2, "center");
    }
  }

  // ---------------------------------------------------------------------
  // phaseWheel — boot progress + legend. `progress` in [0,1]; when it
  // reaches 1 the caller should also set data-state="ready" on the element
  // (CSS switches from the sweep fill to the static legend ramp).
  // ---------------------------------------------------------------------

  function phaseWheel(el, progress) {
    if (!el) return;
    const clamped = Math.max(0, Math.min(1, progress));
    el.style.setProperty("--boot-progress", String(clamped));
  }

  // ---------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------

  window.Render = {
    rampColor,
    rampColorRGB,
    invalidateRamp,
    fitCanvas,
    ensureFit,
    watchDPR,
    drawSpectrum,
    drawWave,
    drawReconstruction,
    drawGrid,
    drawConvolution,
    phaseWheel,
  };
})();
