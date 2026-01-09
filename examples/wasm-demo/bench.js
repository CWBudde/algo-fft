(() => {
  const statusEl = document.getElementById("status");
  const runButton = document.getElementById("run-bench");
  const resultsBody = document.getElementById("results-body");

  let wasmReady = false;

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

  function formatTime(ns) {
    if (ns < 1000) return `${ns.toFixed(0)} ns`;
    return `${(ns / 1000).toFixed(2)} µs`;
  }

  function formatOps(ns) {
    const ops = 1e9 / ns;
    if (ops > 1e6) return `${(ops / 1e6).toFixed(2)} M/s`;
    if (ops > 1e3) return `${(ops / 1e3).toFixed(2)} k/s`;
    return `${ops.toFixed(0)} /s`;
  }

  function formatThroughput(size, ns) {
    const bytesPerOp = size * 16; // 8 bytes input + 8 bytes output
    const opsPerSec = 1e9 / ns;
    const bytesPerSec = bytesPerOp * opsPerSec;
    const mbPerSec = bytesPerSec / 1e6;
    return `${mbPerSec.toFixed(2)} MB/s`;
  }

  async function runBenchmark() {
    if (!wasmReady) return;

    runButton.disabled = true;
    runButton.textContent = "Running...";
    resultsBody.innerHTML = `<tr><td colspan="4" style="text-align: center;">Running benchmark...</td></tr>`;

    // Yield to UI to let it update
    await new Promise((resolve) => setTimeout(resolve, 50));

    try {
        const sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        
        const results = window.algofftBenchmark({
            sizes: sizes,
            minTime: 500
        });

        resultsBody.innerHTML = "";
        
        results.forEach(res => {
            if (res.error) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="4" style="color: #d36b34;">Error for N=${res.size}: ${res.error}</td>`;
                resultsBody.appendChild(tr);
                return;
            }

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${res.size}</td>
                <td>${formatTime(res.avgNs)}</td>
                <td>${formatOps(res.avgNs)}</td>
                <td>${formatThroughput(res.size, res.avgNs)}</td>
            `;
            resultsBody.appendChild(tr);
        });

    } catch (err) {
        console.error(err);
        setStatus("Benchmark failed: " + err.message, false);
    } finally {
        runButton.disabled = false;
        runButton.textContent = "Run Benchmark";
    }
  }

  async function initWasm() {
    if (!WebAssembly.instantiateStreaming) {
      WebAssembly.instantiateStreaming = async (resp, importObject) => {
        const source = await (await resp).arrayBuffer();
        return WebAssembly.instantiate(source, importObject);
      };
    }

    const go = new Go();
    const response = await fetch("algofft.wasm");
    const result = await WebAssembly.instantiateStreaming(
      response,
      go.importObject
    );
    go.run(result.instance);
  }

  initWasm()
    .then(() => {
      wasmReady = true;
      setStatus("WASM ready", true);
      runButton.addEventListener("click", runBenchmark);
    })
    .catch((err) => {
      console.error(err);
      setStatus("WASM failed to load", false);
    });
})();
