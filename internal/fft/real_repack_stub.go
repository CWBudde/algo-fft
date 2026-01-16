//go:build (!amd64 && !arm64) || purego || !asm

package fft

func inverseRepackComplex64SIMD(dst, src, weight []complex64) int {
	return 1
}

func inverseRepackComplex128SIMD(dst, src, weight []complex128) int {
	return 1
}
