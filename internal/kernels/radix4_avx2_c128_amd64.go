//go:build amd64 && !purego

package kernels

import (
	"math"

	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// Radix-4 DIT, 256-bit wide, complex128.
//
// The complex128 side of radix4_avx2_amd64.go, backed by
// internal/asm/amd64/avx2_f64_radix4.s. Shape handling (radix4AVX2Limit) and
// the stage-1 permutation table (radix4GroupIndices) are shared with the
// complex64 kernel: both are properties of n alone, not of the element type.
//
// The twiddle-plane layout is the same as well -- three contiguous planes
// (w1, w2, w3) of m elements per radix-4 stage, plus n/2 for the radix-2 tail
// when n is not a power of four -- so only the element type of the table
// differs. It is generated here rather than converted from the complex64 table
// so the factors are computed in double precision throughout.
//
// A YMM register holds two complex128 rather than four complex64, so the
// kernel retires half as many butterflies per instruction. Everything else,
// including the fused permutation and the folded 1/n, carries over unchanged.

// twiddleSizeRadix4AVX2Complex128 returns the element count of the packed
// twiddle table.
//
// Only the first n-4 elements carry data; the request is padded to n+4 for the
// same reason as in the complex64 kernel, so that a caller handing this codelet
// the plain length-n DIT table is rejected by the kernel's length check instead
// of transforming against the wrong factors.
func twiddleSizeRadix4AVX2Complex128(n int) int {
	if !radix4AVX2SizeOK(n) {
		return 0
	}

	return n + 4
}

// prepareTwiddleRadix4AVX2Complex128 fills dst with the per-stage twiddle
// planes. For the inverse transform the imaginary parts are negated, which is
// the conjugate W^-k = conj(W^k).
func prepareTwiddleRadix4AVX2Complex128(n int, inverse bool, dst []complex128) {
	limit, ok := radix4AVX2Limit(n)
	if !ok || len(dst) < twiddleSizeRadix4AVX2Complex128(n) {
		return
	}

	clear(dst[:twiddleSizeRadix4AVX2Complex128(n)])

	sign := -1.0
	if inverse {
		sign = 1.0
	}

	w := func(e int) complex128 {
		sin, cos := math.Sincos(sign * 2 * math.Pi * float64(e%n) / float64(n))

		return complex(cos, sin)
	}

	offset := 0

	for stage := 4; stage*4 <= limit; stage *= 4 {
		step := n / (4 * stage)

		for mul := 1; mul <= 3; mul++ {
			for j := range stage {
				dst[offset+j] = w(mul * j * step)
			}

			offset += stage
		}
	}

	// Radix-2 tail: one twiddle per butterfly, W_n^j for j = 0..n/2-1.
	if limit != n {
		for j := range n / 2 {
			dst[offset+j] = w(j)
		}
	}
}

// forwardRadix4AVX2Complex128 is the CodeletFunc entry point for the forward
// transform. twiddle must be the table produced by
// prepareTwiddleRadix4AVX2Complex128.
func forwardRadix4AVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex128Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, false, 1)
}

// inverseRadix4AVX2Complex128 is the CodeletFunc entry point for the inverse
// transform. The 1/n normalisation is exact here (n is a power of two) and is
// folded into stage 1 rather than costing a separate pass over the data.
func inverseRadix4AVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	scale := float64(1) / float64(n)

	return amd64.Radix4Complex128Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, true, scale)
}
