//go:build !amd64 || purego

package fft

func recombineForwardComplex64SIMD(dst, src, weight []complex64) int {
	return 1
}

func recombineForwardComplex128SIMD(dst, src, weight []complex128) int {
	return 1
}
