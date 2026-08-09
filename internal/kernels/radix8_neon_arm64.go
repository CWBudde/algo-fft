//go:build arm64 && !purego

package kernels

import "github.com/cwbudde/algo-fft/internal/asm/arm64"

const radix8NEONMinSize = 64

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
