//go:build amd64 && !purego

package math

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

// TransposeSquareOutOfPlaceComplex64 dispatches to the AVX2 64×64/128×128
// transpose asm when n matches one of those sizes and AVX2 is available,
// falling back to the pure-Go TransposeSquareOutOfPlace otherwise. dst and
// src must each hold at least n*n elements and must not alias.
func TransposeSquareOutOfPlaceComplex64(dst, src []complex64, n int) {
	if tryTransposeOutOfPlaceComplex64AVX2(dst, src, n) {
		return
	}

	TransposeSquareOutOfPlace(dst, src, n)
}

// tryTransposeOutOfPlaceComplex64AVX2 attempts the AVX2 asm path for n=64/128
// and reports whether it handled the transform.
func tryTransposeOutOfPlaceComplex64AVX2(dst, src []complex64, n int) bool {
	if (n != 64 && n != 128) || !cpu.DetectFeatures().HasAVX2 {
		return false
	}

	if n == 64 {
		return amd64.Transpose64x64Complex64AVX2Asm(dst, src)
	}

	return amd64.Transpose128x128Complex64AVX2Asm(dst, src)
}

// TransposeSquareTwiddleComplex64Dispatch dispatches to the fused AVX2
// transpose+twiddle asm when n matches 64 or 128 and AVX2 is available,
// falling back to TransposeSquareTwiddleComplex64 otherwise.
func TransposeSquareTwiddleComplex64Dispatch(dst, src, twiddle []complex64, n int) {
	if tryTransposeTwiddleComplex64AVX2(dst, src, twiddle, n) {
		return
	}

	TransposeSquareTwiddleComplex64(dst, src, twiddle, n)
}

// tryTransposeTwiddleComplex64AVX2 attempts the fused AVX2 transpose+twiddle
// asm path for n=64/128 and reports whether it handled the transform.
func tryTransposeTwiddleComplex64AVX2(dst, src, twiddle []complex64, n int) bool {
	if (n != 64 && n != 128) || !cpu.DetectFeatures().HasAVX2 {
		return false
	}

	if n == 64 {
		return amd64.TransposeTwiddle64x64Complex64AVX2Asm(dst, src, twiddle)
	}

	return amd64.TransposeTwiddle128x128Complex64AVX2Asm(dst, src, twiddle)
}

// TransposeSquareTwiddleConjComplex64Dispatch dispatches to the fused AVX2
// transpose+conjugate-twiddle asm when n matches 64 or 128 and AVX2 is
// available, falling back to TransposeSquareTwiddleConjComplex64 otherwise.
func TransposeSquareTwiddleConjComplex64Dispatch(dst, src, twiddle []complex64, n int) {
	if tryTransposeTwiddleConjComplex64AVX2(dst, src, twiddle, n) {
		return
	}

	TransposeSquareTwiddleConjComplex64(dst, src, twiddle, n)
}

// tryTransposeTwiddleConjComplex64AVX2 attempts the fused AVX2
// transpose+conjugate-twiddle asm path for n=64/128 and reports whether it
// handled the transform.
func tryTransposeTwiddleConjComplex64AVX2(dst, src, twiddle []complex64, n int) bool {
	if (n != 64 && n != 128) || !cpu.DetectFeatures().HasAVX2 {
		return false
	}

	if n == 64 {
		return amd64.TransposeTwiddleConj64x64Complex64AVX2Asm(dst, src, twiddle)
	}

	return amd64.TransposeTwiddleConj128x128Complex64AVX2Asm(dst, src, twiddle)
}
