//go:build amd64 && fft_asm && !purego

package kernels

func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Try AVX2 assembly kernel (requires n >= 16, power of 2)
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Try AVX2 assembly kernel (requires n >= 16, power of 2)
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardSSE2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseSSE2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}
