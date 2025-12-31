//go:build arm64 && fft_asm && !purego

package fft

import kasm "github.com/MeKo-Christian/algo-fft/internal/kernels/asm"

func forwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func neonComplexMul2Asm(dst, a, b *complex64) {
	kasm.NeonComplexMul2Asm(dst, a, b)
}

func forwardNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardNEONComplex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseNEONComplex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize32Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize64Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize64Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT4Radix4Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT4Radix4Complex128(dst, src, twiddle, scratch, bitrev)
}
