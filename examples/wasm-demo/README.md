# algofft WASM Demo

An interactive signal lab and benchmark explorer for algofft, compiled to
WebAssembly. Published to GitHub Pages from `main` by
`.github/workflows/wasm-demo-pages.yaml`.

The demo's organising idea is that an FFT bin has a **phase** as well as a
magnitude. Most spectrum plots throw the phase away; this one encodes it as hue
across every visualization.

## Build and run

```bash
just run-wasm-demo    # builds into ./dist and serves it at http://localhost:8090
```

Or build only:

```bash
just build-wasm-demo            # -> ./dist
./scripts/build-wasm-demo.sh /tmp/somewhere
```

The build script copies `*.html`, `*.css`, `*.js` and `*.svg` from this
directory by glob, so new assets ship without editing it.

**An HTTP server is required.** The page fetches `algofft.wasm`, and the
benchmark page runs its transforms in a Web Worker; neither works from a
`file://` URL.

## What it exercises

- Forward and inverse transforms at **complex64 and complex128**
- **Arbitrary lengths**, not just powers of two — the size input accepts any
  `n` from 2 to 2^20. Try 1024 (radix), 1000 (mixed-radix), 1009 (Bluestein) and
  17 (Rader) and watch the reported algorithm change.
- Kernel strategy and planner mode selection, reporting the **resolved** route
  next to the requested one — a forced strategy the dispatch cannot honour at a
  given length is resolved to whatever actually runs, not echoed back
- Window functions (rect / Hann / Hamming / Blackman) and the spectral leakage
  they control
- Round-trip reconstruction error, side by side at both precisions
- 2D transforms, and FFT-based convolution against a naive reference

## Reading the numbers

Go's `js/wasm` target emits **no SIMD and no threads**, so these timings
characterize the portable Go path — not algofft's AVX2/NEON kernels. Expect the
native build to be substantially faster.

Timing is also coarser than you may expect. Under wasm `time.Now()` is
`performance.now()`, which browsers clamp to 100 µs (Chrome) or about 1 ms
(Firefox, Safari) unless the page is cross-origin isolated — which GitHub Pages
cannot be, since it cannot send COOP/COEP headers. A single small transform is
far below one clock tick, so every measurement here is a batch, and the probed
granularity is displayed on the benchmark page. For the same reason the
measuring planner modes (`measure`, `patient`, `exhaustive`) cannot rank kernels
meaningfully at small sizes in a browser.

## Layout

| File                       | Role                                                                 |
| -------------------------- | -------------------------------------------------------------------- |
| `main.go`                  | export table, `globalThis.algofft` assembly                          |
| `bridge.go`                | `guard()` — argument checks and panic recovery around every export   |
| `marshal.go`               | `Float32Array` transfer via `js.CopyBytesToJS` into reusable buffers |
| `plancache.go`             | composite-key LRU plan cache and scratch buffers                     |
| `analyze.go`               | the per-frame transform entry point                                  |
| `signals.go`, `windows.go` | signal generators and window functions                               |
| `info.go`                  | capability tables, derived from the library's own enums              |
| `bench.go`                 | benchmark primitives                                                 |
| `app.js`, `render.js`      | UI wiring and canvas rendering                                       |
| `bench.js`                 | benchmark page controller                                            |

`main_stub.go` keeps the package building on non-wasm targets.

Every export is wrapped in `guard()`, which recovers panics. This matters: a Go
panic under `js/wasm` aborts the instance permanently, so an unguarded edge case
would brick the page rather than fail one call.
