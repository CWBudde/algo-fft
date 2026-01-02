# internal/asm

This directory contains **assembly infrastructure** and contribution guidelines for `algofft`.

## Build tags

- `amd64`, `arm64`: architecture-specific builds.
- `purego`: forces pure-Go builds and **disables any assembly**.
- `asm`: enables assembly-backed kernels in `internal/fft` (when available).

Notes:

- `purego` is treated as an override. If you build with both `-tags=purego,asm`, the `purego` path wins.

## `go:noescape` usage

Assembly entry points should be declared in Go as **externs** (no body) and annotated with `//go:noescape` to help the compiler keep pointer arguments on the stack.

Example pattern:

```go
//go:noescape
func myAsmKernel(dst, src *complex64)
```

## Assembly file conventions

- Keep `.s` files in the **same package** as their Go declarations.
- Use `#include "textflag.h"` and Go assembler syntax.
- Prefer `NOSPLIT|NOFRAME|ABIInternal` for leaf kernels when appropriate.

## Stubs

This package includes a minimal `stubAsm` symbol per-arch to ensure the build-tag and wrapper patterns stay healthy.
