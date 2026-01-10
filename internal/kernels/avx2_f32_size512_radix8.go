//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// forwardAVX2Size512Radix8Complex64 computes a 512-point forward FFT using
// AVX2-accelerated Radix-8 algorithm.
func forwardAVX2Size512Radix8Complex64(dst, src, twiddle, scratch []complex64) bool {
	return amd64.ForwardAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch)
}

// inverseAVX2Size512Radix8Complex64 computes a 512-point inverse FFT using
// AVX2-accelerated Radix-8 algorithm.
func inverseAVX2Size512Radix8Complex64(dst, src, twiddle, scratch []complex64) bool {
	return amd64.InverseAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch)
}
