//go:build asm

package transform

// stockhamPackedEnabled is false under -tags asm: the hand-written SIMD codelet
// path in plan.go is checked before packed Stockham and supersedes it, so the
// pure-Go packed radix-4 route (stockham_packed.go) is redundant here and is
// disabled to keep it out of the SIMD build. Disabling is a dispatch choice, not
// a correctness workaround — the packed path is correct in both builds. See the
// non-asm counterpart in stockham_packed_toggle_go.go.
const stockhamPackedEnabled = false
