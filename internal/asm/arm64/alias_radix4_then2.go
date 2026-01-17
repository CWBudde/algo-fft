//go:build arm64 && asm && !purego

package arm64

// Aliases for radix-4-then-2 kernels (backed by mixed-radix asm symbols).

func ForwardNEONSize32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseNEONSize32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardNEONSize128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseNEONSize128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch)
}
