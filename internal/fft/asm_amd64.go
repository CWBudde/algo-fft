//go:build amd64 && asm && !purego

package fft

import (
	kasm "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
)

func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT4Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT8Radix2Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(8)
	}
	return kasm.ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT64Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT4Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT8Radix2Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(8)
	}
	return kasm.InverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT64Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT4Radix4Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT4Radix4Complex128(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(8)
	}
	return forwardDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(8)
	}
	return inverseDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
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
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch resolveKernelStrategy(len(src), KernelAuto) {
	case KernelDIT:
		return forwardDITComplex128(dst, src, twiddle, scratch, bitrev)
	case KernelStockham:
		return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch resolveKernelStrategy(len(src), KernelAuto) {
	case KernelDIT:
		return inverseDITComplex128(dst, src, twiddle, scratch, bitrev)
	case KernelStockham:
		return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}
