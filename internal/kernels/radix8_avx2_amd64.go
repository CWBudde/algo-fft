//go:build amd64 && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// Radix-8 DIT, 256-bit wide.
//
// The assembly side of radix8_generic.go, backed by
// internal/asm/amd64/avx2_f32_radix8.s. Everything that is a property of n
// alone is shared with the pure-Go ladder: the shape classifier
// (radix8Limit), the packed twiddle layout (twiddleSizeRadix8,
// prepareTwiddleRadix8Complex64) and the stage-1 permutation table
// (radix8GroupIndices). That is the point of having written the prototype
// arch-neutral -- the two kernels cannot disagree about the layout, and the
// generic ladder is the reference the assembly is tested against.
//
// The only thing the assembly adds is a floor on n. Stage 1 retires four
// groups per iteration, so it needs n/8 >= 4; below that the per-size codelets
// own the range anyway.
const radix8AVX2MinSize = 32

// radix8AVX2Limit reports the radix-8 stage limit for n and whether the
// assembly kernel can handle that length at all.
func radix8AVX2Limit(n int) (limit int, ok bool) {
	if n < radix8AVX2MinSize {
		return 0, false
	}

	limit, _, ok = radix8Limit(n)

	return limit, ok
}

// forwardRadix8AVX2Complex64 is the CodeletFunc entry point for the forward
// transform. twiddle must be the table produced by
// prepareTwiddleRadix8Complex64.
func forwardRadix8AVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	limit, ok := radix8AVX2Limit(len(src))
	if !ok {
		return false
	}

	return amd64.Radix8Complex64Asm(dst, src, twiddle, scratch, radix8GroupIndices(len(src)), limit, false, 1)
}

// inverseRadix8AVX2Complex64 is the CodeletFunc entry point for the inverse
// transform. The 1/n normalisation is exact here (n is a power of two) and is
// folded into stage 1 rather than costing a separate pass over the data.
func inverseRadix8AVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix8AVX2Limit(n)
	if !ok {
		return false
	}

	scale := float32(1) / float32(n)

	return amd64.Radix8Complex64Asm(dst, src, twiddle, scratch, radix8GroupIndices(n), limit, true, scale)
}

// forwardRadix8AVX2Complex128 is the complex128 twin of
// forwardRadix8AVX2Complex64, backed by avx2_f64_radix8.s.
func forwardRadix8AVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	limit, ok := radix8AVX2Limit(len(src))
	if !ok {
		return false
	}

	return amd64.Radix8Complex128Asm(dst, src, twiddle, scratch, radix8GroupIndices(len(src)), limit, false, 1)
}

// inverseRadix8AVX2Complex128 is the complex128 twin of
// inverseRadix8AVX2Complex64.
func inverseRadix8AVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix8AVX2Limit(n)
	if !ok {
		return false
	}

	scale := float64(1) / float64(n)

	return amd64.Radix8Complex128Asm(dst, src, twiddle, scratch, radix8GroupIndices(n), limit, true, scale)
}
