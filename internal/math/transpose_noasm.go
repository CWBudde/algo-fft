//go:build !amd64 || purego

package math

// TransposeSquareOutOfPlaceComplex64 is the pure-Go fallback used on builds
// without the amd64 AVX2 transpose asm; see transpose_amd64.go for the SIMD
// dispatch.
func TransposeSquareOutOfPlaceComplex64(dst, src []complex64, n int) {
	TransposeSquareOutOfPlace(dst, src, n)
}

// TransposeSquareTwiddleComplex64Dispatch is the pure-Go fallback used on
// builds without the amd64 AVX2 transpose asm; see transpose_amd64.go for
// the SIMD dispatch.
func TransposeSquareTwiddleComplex64Dispatch(dst, src, twiddle []complex64, n int) {
	TransposeSquareTwiddleComplex64(dst, src, twiddle, n)
}

// TransposeSquareTwiddleConjComplex64Dispatch is the pure-Go fallback used
// on builds without the amd64 AVX2 transpose asm; see transpose_amd64.go
// for the SIMD dispatch.
func TransposeSquareTwiddleConjComplex64Dispatch(dst, src, twiddle []complex64, n int) {
	TransposeSquareTwiddleConjComplex64(dst, src, twiddle, n)
}
