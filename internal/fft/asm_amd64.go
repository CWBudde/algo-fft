//go:build amd64 && !purego

package fft

import (
	kasm "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
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
	return kasm.ForwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}

func inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
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

// forwardAVX2Complex128Asm and inverseAVX2Complex128Asm mirror the complex64
// dispatch order: try radix-4 first (power-of-4 lengths get the pure radix-4
// kernel, other power-of-2 lengths get the radix-4-then-2 "mixed" kernel),
// and only fall back to the radix-2 kernel for lengths the radix-4 kernels
// reject. Radix-4 halves the number of butterfly passes versus radix-2
// (log2(n)/2 vs log2(n)), so this preamble is a straight throughput win
// wherever the radix-4 kernels accept the length; radix-2 remains correct
// (if slower) for everything else.
func forwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	if len(src) >= 64 && m.IsPowerOf4(len(src)) {
		if forwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
			return true
		}
	}
	if len(src) >= 64 && m.IsPowerOf2(len(src)) {
		if forwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
			return true
		}
	}
	return kasm.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}

// inverseAVX2Complex128Asm's radix-4 kernels (unlike their radix-2 and c64
// counterparts) do not apply the 1/n inverse-FFT normalization themselves, so
// this wrapper scales dst explicitly on those two paths before returning.
func inverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if n >= 64 && m.IsPowerOf4(n) {
		if inverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
			ScaleComplex128InPlace(dst, 1.0/float64(n))
			return true
		}
	}
	if n >= 64 && m.IsPowerOf2(n) {
		if inverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
			ScaleComplex128InPlace(dst, 1.0/float64(n))
			return true
		}
	}
	return kasm.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
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

func forwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
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

func inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
}

func forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
}

func inverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
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

func forwardSSE3Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size1024Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size1024Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size4096Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size4096Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
}

func forwardSSE3Size16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.ForwardSSE3Size16384Radix4Complex64Asm(dst, src, twiddle, scratch)
}

func inverseSSE3Size16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return kasm.InverseSSE3Size16384Radix4Complex64Asm(dst, src, twiddle, scratch)
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
	if kasm.ForwardAVX2StockhamComplex128Asm(dst, src, twiddle, scratch) {
		return true
	}
	// The asm kernel declines n < 16; keep those on the Go Stockham path.
	return kernels.ForwardStockhamComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}
	if kasm.InverseAVX2StockhamComplex128Asm(dst, src, twiddle, scratch) {
		return true
	}
	// The asm kernel declines n < 16; keep those on the Go Stockham path.
	return kernels.InverseStockhamComplex128(dst, src, twiddle, scratch)
}

// forwardSSE2Complex128/inverseSSE2Complex128 are the scalar SSE2-complex128 entry
// points. The SSE2 asm kernel only covers a few small sizes, so they fall back to the
// best scalar kernel for the size via the pure size heuristic (planner.ResolveKernelStrategy
// no longer reads any process-global state), keeping them strategy-snapshot-safe.
func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategy(len(src)) {
	case fftypes.KernelStockham:
		return kernels.ForwardStockhamComplex128(dst, src, twiddle, scratch)
	default:
		return kernels.ForwardDITComplex128(dst, src, twiddle, scratch)
	}
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategy(len(src)) {
	case fftypes.KernelStockham:
		return kernels.InverseStockhamComplex128(dst, src, twiddle, scratch)
	default:
		return kernels.InverseDITComplex128(dst, src, twiddle, scratch)
	}
}
