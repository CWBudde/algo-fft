//go:build amd64 && fft_asm && !purego

package fft

// forwardDIT32MixedRadix24Complex64AVX2 is an AVX2-optimized wrapper for size 32 mixed radix FFT.
// It delegates to the proven AVX2 radix-2 implementation.
func forwardDIT32MixedRadix24Complex64AVX2(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT32MixedRadix24Complex64AVX2 is an AVX2-optimized wrapper for size 32 mixed radix inverse FFT.
// It delegates to the proven AVX2 radix-2 inverse implementation.
func inverseDIT32MixedRadix24Complex64AVX2(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

// forwardDIT32MixedRadix24Complex128AVX2 is an AVX2-optimized wrapper for size 32 mixed radix FFT (complex128).
// It delegates to the proven AVX2 radix-2 implementation.
func forwardDIT32MixedRadix24Complex128AVX2(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT32MixedRadix24Complex128AVX2 is an AVX2-optimized wrapper for size 32 mixed radix inverse FFT (complex128).
// It delegates to the proven AVX2 radix-2 inverse implementation.
func inverseDIT32MixedRadix24Complex128AVX2(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev)
}
