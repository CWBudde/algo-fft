//go:build (!amd64 && !arm64) || purego || !asm

package fft

// SIMD stubs for platforms without optimized implementations or when assembly is disabled.
// These always return false, causing fallback to generic implementations.

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
