//go:build 386 && asm && !purego

package fft

import (
	"fmt"
	"unsafe"

	kasm "github.com/cwbudde/algo-fft/internal/asm/x86"
)

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

func forwardSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSESize2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSESize2Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSESize2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSESize2Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSESize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSESize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSESize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSESize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSESize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSESize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSESize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSESize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
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
	if len(bitrev) == 0 {
		return false
	}
	if unsafe.SliceData(bitrev) == nil {
		fmt.Printf("DEBUG: bitrev ptr is nil! len=%d\n", len(bitrev))
	}
	return kasm.ForwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if len(bitrev) == 0 {
		return false
	}
	return kasm.InverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}
