//go:build arm64 && asm && !purego

package fft

import kasm "github.com/MeKo-Christian/algo-fft/internal/asm/arm64"

func forwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
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
	return kasm.ForwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch)
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

func forwardNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Fall back to the Go radix-2 DIT kernel for correctness.
	return forwardDIT32Complex64(dst, src, twiddle, scratch, ComputeBitReversalIndices(32))
}

func inverseNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Fall back to the Go radix-2 DIT kernel for correctness.
	return inverseDIT32Complex64(dst, src, twiddle, scratch, ComputeBitReversalIndices(32))
}

func forwardNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Fall back to the Go radix-2 DIT kernel for correctness.
	return forwardDIT128Complex64(dst, src, twiddle, scratch)
}

func inverseNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	// Fall back to the Go radix-2 DIT kernel for correctness.
	return inverseDIT128Complex64(dst, src, twiddle, scratch)
}

func forwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardNEONSize16Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseNEONSize16Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONSize32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardNEONSize32Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseNEONSize32Complex128Asm(dst, src, twiddle, scratch, bitrev)
}
