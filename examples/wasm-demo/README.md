# algoforge WASM Demo

This demo renders a synthetic audio waveform and its FFT spectrum using
algoforge running in WebAssembly.

## Build

```bash
GOOS=js GOARCH=wasm go build -o algoforge.wasm .
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" .
```

## Run locally

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Notes

- WASM does not use SIMD yet, so performance is lower than native.
- The demo uses `complex64` for speed and limits FFT sizes to 4096.
- Audio playback loops a short buffer to keep the example minimal.
- CI publishes the demo to GitHub Pages via `.github/workflows/wasm-demo-pages.yaml`.
