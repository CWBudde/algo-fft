(() => {
  const statusEl = document.getElementById("status");
  const waveCanvas = document.getElementById("wave");
  const spectrumCanvas = document.getElementById("spectrum");
  const gridCanvas = document.getElementById("grid");
  const sizeSelect = document.getElementById("size");
  const gridSizeSelect = document.getElementById("gridSize");
  const freqAInput = document.getElementById("freqA");
  const freqBInput = document.getElementById("freqB");
  const noiseInput = document.getElementById("noise");
  const animateInput = document.getElementById("animate");
  const playButton = document.getElementById("play");
  const randomizeButton = document.getElementById("randomize");

  const valueSpans = document.querySelectorAll("[data-value]");
  const valueByKey = {};
  valueSpans.forEach((span) => {
    valueByKey[span.dataset.value] = span;
  });

  const state = {
    n: Number(sizeSelect.value),
    gridSize: Number(gridSizeSelect.value),
    freqA: Number(freqAInput.value),
    freqB: Number(freqBInput.value),
    noise: Number(noiseInput.value),
    phase: 0,
    animate: animateInput.checked,
    playing: false,
  };

  let audioCtx = null;
  let audioSource = null;
  let audioGain = null;
  let wasmReady = false;
  let gridOffscreen = null;
  let gridOffscreenCtx = null;
  let gridOffscreenSize = 0;

  const dpr = window.devicePixelRatio || 1;
  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
  }

  function updateValue(key, value) {
    if (!valueByKey[key]) return;
    valueByKey[key].textContent = typeof value === "number" ? value.toString() : value;
  }

  function setStatus(text, ok = true) {
    statusEl.textContent = text;
    statusEl.style.background = ok
      ? "rgba(12, 123, 110, 0.12)"
      : "rgba(211, 107, 52, 0.12)";
    statusEl.style.color = ok ? "#0a5c52" : "#a3471f";
    statusEl.style.borderColor = ok
      ? "rgba(12, 123, 110, 0.25)"
      : "rgba(211, 107, 52, 0.3)";
  }

  function drawWave(signal) {
    const ctx = waveCanvas.getContext("2d");
    ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);

    const midY = waveCanvas.height / 2;
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = "#0c7b6e";

    ctx.beginPath();
    for (let i = 0; i < signal.length; i += 1) {
      const x = (i / (signal.length - 1)) * waveCanvas.width;
      const y = midY - signal[i] * (waveCanvas.height * 0.35);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = "rgba(30, 26, 18, 0.2)";
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(waveCanvas.width, midY);
    ctx.stroke();
  }

  function drawSpectrum(mags) {
    const ctx = spectrumCanvas.getContext("2d");
    ctx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

    const max = mags.reduce((acc, val) => Math.max(acc, val), 1e-9);
    const barWidth = spectrumCanvas.width / mags.length;

    for (let i = 0; i < mags.length; i += 1) {
      const norm = mags[i] / max;
      const height = norm * spectrumCanvas.height * 0.9;
      const x = i * barWidth;
      const y = spectrumCanvas.height - height;
      ctx.fillStyle = i % 2 === 0 ? "#0a5c52" : "#d36b34";
      ctx.fillRect(x, y, barWidth * 0.9, height);
    }
  }

  function palette(t) {
    const stops = [
      { t: 0, r: 22, g: 20, b: 26 },
      { t: 0.45, r: 12, g: 123, b: 110 },
      { t: 0.7, r: 211, g: 107, b: 52 },
      { t: 1, r: 246, g: 206, b: 140 },
    ];
    for (let i = 1; i < stops.length; i += 1) {
      const prev = stops[i - 1];
      const next = stops[i];
      if (t <= next.t) {
        const span = (t - prev.t) / (next.t - prev.t || 1);
        return {
          r: Math.round(prev.r + (next.r - prev.r) * span),
          g: Math.round(prev.g + (next.g - prev.g) * span),
          b: Math.round(prev.b + (next.b - prev.b) * span),
        };
      }
    }
    return { r: 246, g: 206, b: 140 };
  }

  function drawGridSpectrum(mags, size) {
    if (!mags || !size) return;
    const ctx = gridCanvas.getContext("2d");

    if (!gridOffscreen || gridOffscreenSize !== size) {
      gridOffscreen = document.createElement("canvas");
      gridOffscreen.width = size;
      gridOffscreen.height = size;
      gridOffscreenCtx = gridOffscreen.getContext("2d");
      gridOffscreenSize = size;
    }

    const logMags = new Float32Array(mags.length);
    let maxLog = -Infinity;
    for (let i = 0; i < mags.length; i += 1) {
      const value = Math.log10(mags[i] + 1e-6);
      logMags[i] = value;
      if (value > maxLog) maxLog = value;
    }
    if (!Number.isFinite(maxLog) || maxLog <= 0) {
      maxLog = 1;
    }

    const image = gridOffscreenCtx.createImageData(size, size);
    const half = Math.floor(size / 2);
    for (let y = 0; y < size; y += 1) {
      const srcY = (y + half) % size;
      for (let x = 0; x < size; x += 1) {
        const srcX = (x + half) % size;
        const srcIndex = srcY * size + srcX;
        const value = Math.max(0, logMags[srcIndex] / maxLog);
        const color = palette(value);
        const idx = (y * size + x) * 4;
        image.data[idx] = color.r;
        image.data[idx + 1] = color.g;
        image.data[idx + 2] = color.b;
        image.data[idx + 3] = 255;
      }
    }

    gridOffscreenCtx.putImageData(image, 0, 0);
    ctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(gridOffscreen, 0, 0, gridCanvas.width, gridCanvas.height);
  }

  function computeFrame() {
    if (!wasmReady || typeof window.algoforgeFFT !== "function") return;

    const result = window.algoforgeFFT({
      n: state.n,
      gridSize: state.gridSize,
      freqA: state.freqA,
      freqB: state.freqB,
      noise: state.noise,
      phase: state.phase,
    });

    if (result && result.error) {
      setStatus(result.error, false);
      return;
    }

    drawWave(result.signal);
    drawSpectrum(result.spectrum);
    drawGridSpectrum(result.gridSpectrum, result.gridSize);

    if (state.playing) {
      updateAudio(result.signal);
    }
  }

  function tick() {
    if (state.animate) {
      state.phase += 0.04;
    }
    computeFrame();
    requestAnimationFrame(tick);
  }

  function updateAudio(signal) {
    if (!audioCtx) return;
    if (audioSource) {
      audioSource.stop();
      audioSource.disconnect();
    }

    const buffer = audioCtx.createBuffer(1, signal.length, audioCtx.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < signal.length; i += 1) {
      channel[i] = signal[i] * 0.3;
    }

    audioSource = audioCtx.createBufferSource();
    audioSource.buffer = buffer;
    audioSource.loop = true;
    audioSource.connect(audioGain);
    audioSource.start();
  }

  function toggleAudio() {
    if (!audioCtx) {
      audioCtx = new AudioContext();
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
      return;
    }

    state.playing = true;
    playButton.textContent = "Stop audio";
    computeFrame();
  }

  function randomize() {
    const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    state.freqA = rand(2, 24);
    state.freqB = rand(18, 96);
    state.noise = Math.round((Math.random() * 0.25 + 0.02) * 100) / 100;

    freqAInput.value = state.freqA;
    freqBInput.value = state.freqB;
    noiseInput.value = state.noise;

    updateValue("freqA", state.freqA);
    updateValue("freqB", state.freqB);
    updateValue("noise", state.noise.toFixed(2));
  }

  function handleInput() {
    state.n = Number(sizeSelect.value);
    state.gridSize = Number(gridSizeSelect.value);
    state.freqA = Number(freqAInput.value);
    state.freqB = Number(freqBInput.value);
    state.noise = Number(noiseInput.value);
    state.animate = animateInput.checked;

    updateValue("freqA", state.freqA);
    updateValue("freqB", state.freqB);
    updateValue("noise", state.noise.toFixed(2));

    computeFrame();
  }

  async function initWasm() {
    if (!WebAssembly.instantiateStreaming) {
      WebAssembly.instantiateStreaming = async (resp, importObject) => {
        const source = await (await resp).arrayBuffer();
        return WebAssembly.instantiate(source, importObject);
      };
    }

    const go = new Go();
    const response = await fetch("algoforge.wasm");
    const result = await WebAssembly.instantiateStreaming(response, go.importObject);
    go.run(result.instance);
  }

  function boot() {
    resizeCanvas(waveCanvas);
    resizeCanvas(spectrumCanvas);
    resizeCanvas(gridCanvas);

    window.addEventListener("resize", () => {
      resizeCanvas(waveCanvas);
      resizeCanvas(spectrumCanvas);
      resizeCanvas(gridCanvas);
      computeFrame();
    });

    sizeSelect.addEventListener("change", handleInput);
    gridSizeSelect.addEventListener("change", handleInput);
    freqAInput.addEventListener("input", handleInput);
    freqBInput.addEventListener("input", handleInput);
    noiseInput.addEventListener("input", handleInput);
    animateInput.addEventListener("change", handleInput);
    playButton.addEventListener("click", toggleAudio);
    randomizeButton.addEventListener("click", () => {
      randomize();
      handleInput();
    });

    handleInput();
    requestAnimationFrame(tick);
  }

  initWasm()
    .then(() => {
      wasmReady = true;
      setStatus("WASM ready", true);
      boot();
    })
    .catch((err) => {
      console.error(err);
      setStatus("WASM failed to load", false);
    });
})();
