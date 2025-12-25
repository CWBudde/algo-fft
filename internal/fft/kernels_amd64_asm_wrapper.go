//go:build amd64 && fft_asm && !purego

package fft

func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Try AVX2 assembly kernel (requires n >= 16, power of 2)
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Try AVX2 assembly kernel (requires n >= 16, power of 2)
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	return false
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	return false
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Complex128(dst, src, twiddle, scratch, bitrev)
}
