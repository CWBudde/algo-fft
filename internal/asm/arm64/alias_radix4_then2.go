//go:build arm64 && !purego

package arm64

// Aliases for radix-4-then-2 kernels (backed by mixed-radix asm symbols).
//
// The complex64 32/128 aliases used to live here too, but they now call the
// shared NEON Stockham radix-4 core directly (see neon_radix4_loop.go) rather
// than the retired size-specific mixed-radix asm, so they moved there with
// their same-size Go-only siblings (512, 2048, 8192, 32768).

func ForwardNEONSize32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return ForwardNEONSize32MixedRadix24Complex128Asm(dst, src, twiddle, scratch)
}

func InverseNEONSize32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return InverseNEONSize32MixedRadix24Complex128Asm(dst, src, twiddle, scratch)
}

func ForwardNEONSize128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return ForwardNEONSize128MixedRadix24Complex128Asm(dst, src, twiddle, scratch)
}

func InverseNEONSize128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return InverseNEONSize128MixedRadix24Complex128Asm(dst, src, twiddle, scratch)
}
