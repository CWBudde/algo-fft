//go:build amd64 && !purego

package kernels

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// Codelet bindings for the generic AVX-512 radix-2 DIT kernel
// (internal/asm/amd64/avx512_f32_generic.s).
//
// The kernel handles any power of two >= 16, but it is registered as a
// codelet only at the sizes where it beats the best AVX2 codelet, measured
// on AVX-512 hardware (Xeon 2.8 GHz, Skylake-SP class), forward complex64:
//
//	size    AVX-512    best AVX2 codelet
//	1024    7.9 µs     19.8 µs (radix-32x32)   2.5x
//	2048    16.1 µs    13.9 µs (radix-4+2)     AVX2 codelet kept
//	4096    34.8 µs    41.5 µs (six-step)      1.2x
//	8192    76 µs      147 µs  (six-step)      1.9x
//	16384   179 µs     230 µs  (six-step)      1.3x
//
// The inverse direction shows the same winners. For complex128 the AVX2
// codelets win at every size >= 2048 and tie at 1024, so no complex128
// AVX-512 codelets are registered. Codelet selection prefers a higher SIMD
// level over priority, so an AVX-512 entry always outranks the AVX2 ones on
// AVX-512 hosts — only the winning sizes may be registered.
//
// Sizes without any registered codelet reach the same kernel through the
// dispatch tier in internal/fft (kernels_amd64_avx512.go).

// Radix-2 bit-reversal tables for the registered AVX-512 codelet sizes.
//
//nolint:gochecknoglobals
var (
	bitrevSize1024Radix2  = mathpkg.ComputeBitReversalIndices(1024)
	bitrevSize4096Radix2  = mathpkg.ComputeBitReversalIndices(4096)
	bitrevSize8192Radix2  = mathpkg.ComputeBitReversalIndices(8192)
	bitrevSize16384Radix2 = mathpkg.ComputeBitReversalIndices(16384)
)

// avx512CodeletBitrev returns the precomputed radix-2 bit-reversal table for
// the sizes registered as AVX-512 codelets, or nil for any other size.
func avx512CodeletBitrev(n int) []int {
	switch n {
	case 1024:
		return bitrevSize1024Radix2
	case 4096:
		return bitrevSize4096Radix2
	case 8192:
		return bitrevSize8192Radix2
	case 16384:
		return bitrevSize16384Radix2
	default:
		return nil
	}
}

// forwardAVX512Radix2Complex64 adapts the generic AVX-512 kernel to the
// four-argument codelet signature by supplying the precomputed bit-reversal
// table for the registered sizes.
func forwardAVX512Radix2Complex64(dst, src, twiddle, scratch []complex64) bool {
	bitrev := avx512CodeletBitrev(len(src))
	if bitrev == nil {
		return false
	}

	return amd64.ForwardAVX512Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

// inverseAVX512Radix2Complex64 is the inverse counterpart of
// forwardAVX512Radix2Complex64 (conjugate twiddles, scales by 1/n).
func inverseAVX512Radix2Complex64(dst, src, twiddle, scratch []complex64) bool {
	bitrev := avx512CodeletBitrev(len(src))
	if bitrev == nil {
		return false
	}

	return amd64.InverseAVX512Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
