/*
 * bench-worker.js — the benchmark's wasm instance, off the main thread.
 *
 * CLASSIC worker, deliberately. wasm_exec.js is a classic script that assigns
 * globalThis.Go, and module workers cannot importScripts(), so a
 * type:"module" worker has no way to load it short of rewriting it. The Go
 * runtime itself is fine here: it touches globalThis only, never window or
 * document, and everything it asserts at load (crypto.getRandomValues,
 * performance.now, TextEncoder/TextDecoder, setTimeout) exists in a worker.
 *
 * Two things about the control flow are not obvious and are load-bearing.
 *
 * 1. go.run() must NOT be awaited. The demo's main() ends in select{}, so the
 *    promise go.run() returns never resolves. Awaiting it means never posting
 *    "ready". Instead we start it and poll for self.algofft to appear.
 *
 * 2. A call into Go is synchronous and blocks this worker's event loop for its
 *    whole duration, so a "cancel" message posted by the page cannot be
 *    dispatched while a Go call is in flight. That is why the benchmark loop
 *    lives here rather than in Go: each benchStep is bounded to roughly one
 *    target window (~50-120 ms), and between steps we yield with setTimeout so
 *    the queued cancel actually lands. The page keeps worker.terminate() as
 *    the hard fallback behind a watchdog, but in normal operation this yield
 *    is the whole cancellation mechanism.
 */

/* global Go, importScripts */

importScripts("wasm_exec.js");

const READY_POLL_MS = 10;
const READY_TIMEOUT_MS = 30000;

/** The runId whose cancellation has been requested, or null. */
let cancelRequested = null;
/** The runId currently executing, or null. */
let activeRun = null;

let api = null;

function post(message) {
  self.postMessage(message);
}

/** Yield to the worker event loop so queued messages (cancel) can dispatch. */
function yieldToLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function boot() {
  if (!WebAssembly.instantiateStreaming) {
    WebAssembly.instantiateStreaming = async (resp, importObject) => {
      const source = await (await resp).arrayBuffer();
      return WebAssembly.instantiate(source, importObject);
    };
  }

  const go = new Go();
  const result = await WebAssembly.instantiateStreaming(
    fetch("algofft.wasm"),
    go.importObject,
  );

  // Intentionally not awaited: main() ends in select{} and never returns.
  go.run(result.instance).catch((err) => {
    post({ type: "fatal", error: `go runtime exited: ${err && err.message}` });
  });

  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (!self.algofft) {
    if (Date.now() > deadline) {
      throw new Error("timed out waiting for the wasm module to publish algofft");
    }
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }

  api = self.algofft;

  const info = api.info();
  if (info && info.error) {
    throw new Error(info.error);
  }

  post({ type: "ready", info });
}

/**
 * Runs one full sweep. Every Go call is checked for the {error, panic} shape
 * the bridge's guard() returns; a panicking case is reported and the sweep
 * continues, because a forced strategy at a size it cannot execute is exactly
 * the kind of thing this page is meant to surface.
 */
async function runJob(runId, request) {
  activeRun = runId;

  const started = api.benchStart(request);
  if (!started || started.error) {
    post({
      type: "jobError",
      runId,
      error: (started && started.error) || "benchStart returned nothing",
      fatal: true,
    });
    post({ type: "runDone", runId, completed: 0, cancelled: false });
    activeRun = null;
    return;
  }

  const id = started.id;
  const total = started.total;

  post({
    type: "jobStarted",
    runId,
    id,
    total,
    trials: started.trials,
    targetMs: started.targetMs,
    granularityNs: started.granularityNs,
    cases: started.cases,
  });

  let completed = 0;
  let cancelled = false;

  for (;;) {
    if (cancelRequested === runId) {
      cancelled = true;
      api.benchCancel({ id });
      break;
    }

    const step = api.benchStep({ id });

    if (!step || step.error) {
      post({
        type: "jobError",
        runId,
        error: (step && step.error) || "benchStep returned nothing",
        panic: Boolean(step && step.panic),
        fatal: true,
      });
      break;
    }

    if (step.prepared) {
      post({ type: "jobPrepared", runId, caseIndex: step.caseIndex, prepared: step.prepared });
    }

    if (step.result) {
      completed += 1;
      post({ type: "jobResult", runId, caseIndex: step.caseIndex, result: step.result });
    }

    post({
      type: "jobProgress",
      runId,
      caseIndex: step.caseIndex,
      total: step.total,
      phase: step.phase,
      trial: step.trial || 0,
      trials: step.trials || started.trials,
      iterations: step.iterations || 0,
      completed,
    });

    if (step.done) {
      break;
    }

    // The yield that makes Stop work. Without it the loop above never returns
    // to the event loop and the cancel message waits for the entire sweep.
    await yieldToLoop();
  }

  if (cancelled) {
    // Best effort; the job may already have been dropped by benchStep.
    try {
      api.benchCancel({ id });
    } catch (err) {
      void err;
    }
  }

  post({ type: "runDone", runId, completed, cancelled });
  activeRun = null;
  cancelRequested = null;
}

self.onmessage = (event) => {
  const message = event.data || {};

  switch (message.type) {
    case "cancel":
      cancelRequested = message.runId != null ? message.runId : activeRun;
      break;

    case "run":
      if (!api) {
        post({ type: "jobError", runId: message.runId, error: "wasm not ready", fatal: true });
        post({ type: "runDone", runId: message.runId, completed: 0, cancelled: false });
        return;
      }
      if (activeRun !== null) {
        post({
          type: "jobError",
          runId: message.runId,
          error: "a benchmark is already running in this worker",
          fatal: true,
        });
        post({ type: "runDone", runId: message.runId, completed: 0, cancelled: false });
        return;
      }
      cancelRequested = null;
      runJob(message.runId, message.request).catch((err) => {
        post({ type: "fatal", error: String((err && err.message) || err) });
      });
      break;

    case "release":
      self.close();
      break;

    default:
      break;
  }
};

boot().catch((err) => {
  post({ type: "fatal", error: String((err && err.message) || err) });
});
