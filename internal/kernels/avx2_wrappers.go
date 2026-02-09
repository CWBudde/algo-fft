//go:build amd64 && asm && !purego

package kernels

import amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"

// forwardAVX2Size256Radix4Complex64Safe ensures in-place forward uses dst output.
// The AVX2 radix-4 size-256 forward kernel writes to scratch for in-place use
// but does not copy results back to dst.
func forwardAVX2Size256Radix4Complex64Safe(dst, src, twiddle, scratch []complex64) bool {
	const n = 256
	if len(dst) < n || len(src) < n || len(scratch) < n {
		return amd64.ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
	}

	if &dst[0] == &src[0] {
		ok := amd64.ForwardAVX2Size256Radix4Complex64Asm(scratch, src, twiddle, scratch)
		if ok {
			copy(dst[:n], scratch[:n])
		}
		return ok
	}

	return amd64.ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
}

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

// forwardAVX2Size512Radix16x32Complex64 computes a 512-point forward FFT using
// AVX2-accelerated Radix-16x32 algorithm with Go fallback.
func forwardAVX2Size512Radix16x32Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch) {
		return forwardDIT512Mixed16x32Complex64(dst, src, twiddle, scratch)
	}
	return true
}

// inverseAVX2Size512Radix16x32Complex64 computes a 512-point inverse FFT using
// AVX2-accelerated Radix-16x32 algorithm with Go fallback.
func inverseAVX2Size512Radix16x32Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !amd64.InverseAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch) {
		return inverseDIT512Mixed16x32Complex64(dst, src, twiddle, scratch)
	}
	return true
}
