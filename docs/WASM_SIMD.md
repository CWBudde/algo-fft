# WebAssembly SIMD research (Phase 23.2)

## Summary

- WebAssembly has a standardized fixed-width SIMD proposal and a newer
  "relaxed SIMD" proposal. The feature status page lists fixed-width SIMD
  as broadly supported across major engines, while relaxed SIMD is still
  behind flags in some runtimes.
- Go's official WebAssembly wiki and the Go 1.25 release notes do not
  mention SIMD support for the `js/wasm` port. A quick scan of the Go
  compiler's wasm backend sources in GOROOT does not show SIMD-related
  instructions, suggesting the standard Go toolchain does not currently
  emit WASM SIMD instructions.
- Result: no prototype or benchmark was built because the primary Go
  toolchain appears to lack SIMD codegen for `js/wasm` at this time.

## Sources

- Go WebAssembly wiki: https://go.dev/wiki/WebAssembly
- Go 1.25 release notes: https://go.dev/doc/go1.25
- WebAssembly feature status: https://webassembly.org/features/
- WebAssembly features JSON: https://webassembly.org/features.json

## Next steps (if SIMD becomes available in Go)

- Track Go compiler support for WASM SIMD instructions.
- Add a small butterfly kernel using SIMD when build tags allow.
- Compare performance vs. the current scalar WASM path.
