//go:build amd64 && asm && !purego

package fft

import (
	kasm "github.com/cwbudde/algo-fft/internal/asm/amd64"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	if len(src) >= 64 && m.IsPowerOf4(len(src)) {
		if kasm.ForwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}
	if len(src) >= 64 && m.IsPowerOf2(len(src)) {
		if kasm.ForwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, nil)
}

func inverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	if len(src) >= 64 && m.IsPowerOf4(len(src)) {
		if kasm.InverseAVX2Complex64Radix4Asm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}
	if len(src) >= 64 && m.IsPowerOf2(len(src)) {
		if kasm.InverseAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, nil)
}

func forwardAVX2Complex64GenericRadix2Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, nil)
}

func forwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch, nil)
}

func forwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch, nil)
}

func inverseAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch, nil)
}

func forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, m.ComputeBitReversalIndices(len(src)))
}

func inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, m.ComputeBitReversalIndices(len(src)))
}

func forwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, m.ComputeBitReversalIndices(len(src)))
}

func inverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, m.ComputeBitReversalIndices(len(src)))
}

func forwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch, nil)
}

func inverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch, nil)
}

func forwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch, nil)
}

func inverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch, nil)
}

func forwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size2048Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size2048Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size2048Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size2048Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size128Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size128Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size128Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size128Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return forwardSSE2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return inverseSSE2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return sse3SizeSpecificOrGenericDITComplex64(KernelAuto)(dst, src, twiddle, scratch)
}

func inverseSSE3Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return sse3SizeSpecificOrGenericDITInverseComplex64(KernelAuto)(dst, src, twiddle, scratch)
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if !m.IsPowerOf2(n) {
		return false
	}
	return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if !m.IsPowerOf2(n) {
		return false
	}
	return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return forwardStockhamComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	return inverseStockhamComplex128(dst, src, twiddle, scratch)
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategyWithDefault(len(src), KernelAuto) {
	case KernelDIT:
		return forwardDITComplex128(dst, src, twiddle, scratch)
	case KernelStockham:
		return forwardStockhamComplex128(dst, src, twiddle, scratch)
	default:
		return false
	}
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategyWithDefault(len(src), KernelAuto) {
	case KernelDIT:
		return inverseDITComplex128(dst, src, twiddle, scratch)
	case KernelStockham:
		return inverseStockhamComplex128(dst, src, twiddle, scratch)
	default:
		return false
	}
}
