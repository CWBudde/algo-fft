//go:build amd64 && fft_asm && !purego

package fft

// forwardDIT128MixedRadix24Complex64AVX2 is an AVX2-optimized wrapper for size 128 mixed radix FFT.
// It delegates to the proven AVX2 radix-2 implementation.
func forwardDIT128MixedRadix24Complex64AVX2(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT128MixedRadix24Complex64AVX2 is an AVX2-optimized wrapper for size 128 mixed radix inverse FFT.
// It delegates to the proven AVX2 radix-2 inverse implementation.
func inverseDIT128MixedRadix24Complex64AVX2(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

// forwardDIT128MixedRadix24Complex128AVX2 is a wrapper for size 128 mixed radix FFT (complex128).
// Currently delegates to the pure Go implementation as AVX2 complex128 kernels are not yet
// available for size 128. This maintains consistency with the registered codelet.
func forwardDIT128MixedRadix24Complex128AVX2(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT128MixedRadix24Complex128(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT128MixedRadix24Complex128AVX2 is a wrapper for size 128 mixed radix inverse FFT (complex128).
// Currently delegates to the pure Go implementation as AVX2 complex128 kernels are not yet
// available for size 128. This maintains consistency with the registered codelet.
func inverseDIT128MixedRadix24Complex128AVX2(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT128MixedRadix24Complex128(dst, src, twiddle, scratch, bitrev)
}
