//go:build arm64 && fft_asm && !purego

package asm

// The size-specific dispatcher in internal/fft references a 128-point mixed-radix
// NEON kernel. Until the dedicated asm variant exists, route it to the existing
// size-128 radix-2 implementation.

func forwardNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
