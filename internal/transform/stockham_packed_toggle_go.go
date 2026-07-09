//go:build (!amd64 && !arm64 && !386) || purego

package transform

// stockhamPackedEnabled gates the pure-Go packed radix-4 Stockham fast path
// (see stockhamPacked in stockham_packed.go). It is enabled on pure-Go builds
// (architectures without SIMD codelets, or -tags purego), where it is the
// fastest Stockham route available. On SIMD builds the hand-written codelet
// path in plan.go is checked first and supersedes packed Stockham, so the
// toggle flips to false there (stockham_packed_toggle_simd.go) to keep the
// now-redundant branch out of the SIMD build. This is a dispatch
// de-duplication, not a correctness workaround: the packed path itself is
// correct in both builds.
const stockhamPackedEnabled = true
