//go:build amd64 && (!asm || purego)

package fft

// Fallback stubs for AMD64 when assembly optimizations are disabled.
// These functions always return false, causing the caller to use generic implementations.

func complexMulArrayComplex64SIMD(dst, a, b []complex64) bool {
	return false
}

func complexMulArrayComplex128SIMD(dst, a, b []complex128) bool {
	return false
}

func complexMulArrayInPlaceComplex64SIMD(dst, src []complex64) bool {
	return false
}

func complexMulArrayInPlaceComplex128SIMD(dst, src []complex128) bool {
	return false
}
