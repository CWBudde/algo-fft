//go:build 386 && fft_asm && !purego

package kernels

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
