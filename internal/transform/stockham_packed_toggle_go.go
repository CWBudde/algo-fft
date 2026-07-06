//go:build !asm

package transform

// stockhamPackedEnabled gates the pure-Go packed radix-4 Stockham fast path
// (see stockhamPacked in stockham_packed.go). It is enabled on the default
// (non-asm) build, where it is the fastest Stockham route available. Under
// -tags asm the hand-written SIMD codelet path in plan.go is checked first and
// supersedes packed Stockham, so the toggle flips to false there
// (stockham_packed_toggle_asm.go) to keep the now-redundant branch out of the
// SIMD build. This is a dispatch de-duplication, not a correctness workaround:
// the packed path itself is correct in both builds.
const stockhamPackedEnabled = true
