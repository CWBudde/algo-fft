//go:build 386 && asm && !purego

package fft

import kasm "github.com/MeKo-Christian/algo-fft/internal/asm/x86"

// Wrapper functions that call the x86 assembly implementations
func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSEComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSEComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSEComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSEComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}
