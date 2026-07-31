//go:build amd64 && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// Radix-8 DIT, 512-bit wide.
//
// The AVX-512 sibling of radix8_avx2_amd64.go, and deliberately its shape: the
// shape classifier (radix8Limit), the packed twiddle layout
// (twiddleSizeRadix8, prepareTwiddleRadix8Complex64) and the stage-1
// permutation table (radix8GroupIndices) are the ones the pure-Go ladder
// defines, so no kernel in the tree can disagree with another about the
// layout, and the generic ladder stays the reference all of them are tested
// against.
//
// The width changes only the floor on n. Stage 1 retires one ZMM of groups per
// iteration -- eight for complex64, four for complex128 -- so it needs
// n/8 >= 8 and n/8 >= 4 respectively.
const (
	radix8AVX512MinSize64  = 64
	radix8AVX512MinSize128 = 32
)

// radix8AVX512Limit64 reports the radix-8 stage limit for n and whether the
// 512-bit complex64 kernel can handle that length at all.
func radix8AVX512Limit64(n int) (limit int, ok bool) {
	if n < radix8AVX512MinSize64 {
		return 0, false
	}

	limit, _, ok = radix8Limit(n)

	return limit, ok
}

// radix8AVX512Limit128 is radix8AVX512Limit64 for the complex128 kernel, whose
// stage-1 floor is one ZMM of four groups rather than eight.
func radix8AVX512Limit128(n int) (limit int, ok bool) {
	if n < radix8AVX512MinSize128 {
		return 0, false
	}

	limit, _, ok = radix8Limit(n)

	return limit, ok
}

// forwardRadix8AVX512Complex64 is the CodeletFunc entry point for the forward
// transform. twiddle must be the table produced by
// prepareTwiddleRadix8Complex64.
func forwardRadix8AVX512Complex64(dst, src, twiddle, scratch []complex64) bool {
	limit, ok := radix8AVX512Limit64(len(src))
	if !ok {
		return false
	}

	return amd64.Radix8AVX512Complex64Asm(
		dst, src, twiddle, scratch, radix8GroupIndices(len(src)), limit, false, 1,
	)
}

// inverseRadix8AVX512Complex64 is the CodeletFunc entry point for the inverse
// transform. The 1/n normalisation is exact here (n is a power of two) and is
// folded into stage 1 rather than costing a separate pass over the data.
func inverseRadix8AVX512Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix8AVX512Limit64(n)
	if !ok {
		return false
	}

	scale := float32(1) / float32(n)

	return amd64.Radix8AVX512Complex64Asm(
		dst, src, twiddle, scratch, radix8GroupIndices(n), limit, true, scale,
	)
}

// forwardRadix8AVX512Complex128 is the complex128 twin of
// forwardRadix8AVX512Complex64, backed by avx512_f64_radix8.s.
func forwardRadix8AVX512Complex128(dst, src, twiddle, scratch []complex128) bool {
	limit, ok := radix8AVX512Limit128(len(src))
	if !ok {
		return false
	}

	return amd64.Radix8AVX512Complex128Asm(
		dst, src, twiddle, scratch, radix8GroupIndices(len(src)), limit, false, 1,
	)
}

// inverseRadix8AVX512Complex128 is the complex128 twin of
// inverseRadix8AVX512Complex64.
func inverseRadix8AVX512Complex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix8AVX512Limit128(n)
	if !ok {
		return false
	}

	scale := float64(1) / float64(n)

	return amd64.Radix8AVX512Complex128Asm(
		dst, src, twiddle, scratch, radix8GroupIndices(n), limit, true, scale,
	)
}
