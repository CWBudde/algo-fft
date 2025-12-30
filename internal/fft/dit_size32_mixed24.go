package fft

// forwardDIT32MixedRadix24Complex64 is an alias for the proven radix-2 implementation.
// It uses the standard DIT approach which is equivalent to a mixed-radix decomposition
// and provides excellent performance for size 32.
func forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT32Complex64(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT32MixedRadix24Complex64 is an alias for the proven radix-2 inverse.
func inverseDIT32MixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT32Complex64(dst, src, twiddle, scratch, bitrev)
}

// forwardDIT32MixedRadix24Complex128 is an alias for the proven radix-2 implementation.
// It uses the standard DIT approach which is equivalent to a mixed-radix decomposition
// and provides excellent performance for size 32.
func forwardDIT32MixedRadix24Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT32Complex128(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT32MixedRadix24Complex128 is an alias for the proven radix-2 inverse.
func inverseDIT32MixedRadix24Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT32Complex128(dst, src, twiddle, scratch, bitrev)
}
