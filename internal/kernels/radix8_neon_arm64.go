//go:build arm64 && !purego

package kernels

import "github.com/cwbudde/algo-fft/internal/asm/arm64"

// The assembly handles one radix-8 stage followed by a radix-4 tail at n=32.
const radix8NEONMinSize = 32

const radix8NEONSize32Limit = 8

// The size-32 codelet is hot enough that repeating radix8Limit and the
// sync.Once-backed group-table lookup on every transform is measurable. These
// are the four group bases from ComputeBitReversalIndicesRadix8Then4(32), whose
// complete permutation is [0,4,...,28, 1,5,...,29, 2,6,...,30, 3,7,...,31].
// The assembly still validates every slice.
//
//nolint:gochecknoglobals // immutable permutation for a fixed-size codelet
var radix8NEONSize32Groups = [...]int32{0, 1, 2, 3}

func radix8NEONLimit(n int) (limit int, ok bool) {
	if n < radix8NEONMinSize {
		return 0, false
	}

	limit, _, ok = radix8Limit(n)

	return limit, ok
}

func forwardRadix8NEONComplex64(dst, src, twiddle, scratch []complex64) bool {
	limit, ok := radix8NEONLimit(len(src))
	if !ok {
		return false
	}

	return arm64.Radix8Complex64Asm(dst, src, twiddle, scratch, radix8GroupIndices(len(src)), limit, false, 1)
}

func inverseRadix8NEONComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix8NEONLimit(n)
	if !ok {
		return false
	}

	return arm64.Radix8Complex64Asm(
		dst, src, twiddle, scratch, radix8GroupIndices(n), limit, true, float32(1)/float32(n),
	)
}

func forwardRadix8NEONComplex128(dst, src, twiddle, scratch []complex128) bool {
	limit, ok := radix8NEONLimit(len(src))
	if !ok {
		return false
	}

	return arm64.Radix8Complex128Asm(dst, src, twiddle, scratch, radix8GroupIndices(len(src)), limit, false, 1)
}

func inverseRadix8NEONComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix8NEONLimit(n)
	if !ok {
		return false
	}

	return arm64.Radix8Complex128Asm(
		dst, src, twiddle, scratch, radix8GroupIndices(n), limit, true, float64(1)/float64(n),
	)
}

func forwardRadix8NEONSize32Complex64(dst, src, twiddle, scratch []complex64) bool {
	return arm64.Radix8Complex64Asm(
		dst, src, twiddle, scratch, radix8NEONSize32Groups[:], radix8NEONSize32Limit, false, 1,
	)
}

func inverseRadix8NEONSize32Complex64(dst, src, twiddle, scratch []complex64) bool {
	return arm64.Radix8Complex64Asm(
		dst, src, twiddle, scratch, radix8NEONSize32Groups[:], radix8NEONSize32Limit, true, 1.0/32.0,
	)
}

func forwardRadix8NEONSize32Complex128(dst, src, twiddle, scratch []complex128) bool {
	return arm64.Radix8Complex128Asm(
		dst, src, twiddle, scratch, radix8NEONSize32Groups[:], radix8NEONSize32Limit, false, 1,
	)
}

func inverseRadix8NEONSize32Complex128(dst, src, twiddle, scratch []complex128) bool {
	return arm64.Radix8Complex128Asm(
		dst, src, twiddle, scratch, radix8NEONSize32Groups[:], radix8NEONSize32Limit, true, 1.0/32.0,
	)
}
