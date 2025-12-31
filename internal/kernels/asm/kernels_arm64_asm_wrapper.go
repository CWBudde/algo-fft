//go:build arm64 && fft_asm && !purego

package asm

func ForwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func NeonComplexMul2Asm(dst, a, b *complex64) {
	neonComplexMul2Asm(dst, a, b)
}

func ForwardNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardNEONComplex128Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseNEONComplex128Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize64Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize64Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func ForwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func InverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
