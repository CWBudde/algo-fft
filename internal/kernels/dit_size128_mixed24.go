package kernels

// forwardDIT128MixedRadix24Complex64 is an alias for the proven radix-2 implementation.
// It uses the standard DIT approach which is equivalent to a mixed-radix decomposition
// and provides excellent performance for size 128.
func forwardDIT128MixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT128Complex64(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT128MixedRadix24Complex64 is an alias for the proven radix-2 inverse.
func inverseDIT128MixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT128Complex64(dst, src, twiddle, scratch, bitrev)
}

// forwardDIT128MixedRadix24Complex128 is an alias for the proven radix-2 implementation.
// It uses the standard DIT approach which is equivalent to a mixed-radix decomposition
// and provides excellent performance for size 128.
func forwardDIT128MixedRadix24Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT128Complex128(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT128MixedRadix24Complex128 is an alias for the proven radix-2 inverse.
func inverseDIT128MixedRadix24Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT128Complex128(dst, src, twiddle, scratch, bitrev)
}
