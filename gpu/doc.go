// Package gpu provides an experimental GPU backend for algofft.
//
// This package defines a dedicated GPU plan API that mirrors the CPU plan
// surface while allowing persistent device buffers and backend-specific
// execution contexts. The implementation is intentionally minimal and
// currently requires a backend to be registered at runtime.
package gpu
