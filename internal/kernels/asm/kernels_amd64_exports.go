//go:build amd64 && fft_asm && !purego

package asm

func ForwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardSSE2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseSSE2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev)
}
