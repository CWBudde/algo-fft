//go:build (!amd64 && !arm64) || purego || !asm

package fft

func scaleComplex64SIMD(dst []complex64, scale float32) bool {
	return false
}

func scaleComplex128SIMD(dst []complex128, scale float64) bool {
	return false
}
