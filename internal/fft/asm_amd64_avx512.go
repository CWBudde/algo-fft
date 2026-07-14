//go:build amd64 && !purego

package fft

import (
	kasm "github.com/cwbudde/algo-fft/internal/asm/amd64"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Bridge wrappers for the AVX-512 generic radix-2 DIT kernels. The asm
// kernels validate lengths and fall back (return false) for sizes below 16,
// so callers can chain them in front of the AVX2 kernels. The shared per-size
// bit-reversal table keeps the wrappers allocation-free after the first
// transform of a given size.

func forwardAVX512Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kasm.ForwardAVX512Complex64Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}

func inverseAVX512Complex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kasm.InverseAVX512Complex64Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}

func forwardAVX512Complex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kasm.ForwardAVX512Complex128Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}

func inverseAVX512Complex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kasm.InverseAVX512Complex128Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}
