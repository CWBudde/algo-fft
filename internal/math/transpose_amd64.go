//go:build amd64 && !purego && fftprobe

// Probe-gated AVX2 transposes. The six asm symbols behind this file are
// correct and tested, but nothing in the library calls the three dispatch
// entry points below: the six-step and four-step routes that would use them
// are Phase 3 work. Rather than leave assembly that only a test can reach —
// which is how a wrong kernel survives a green suite — the whole dispatch
// layer sits behind `-tags fftprobe`, so an ordinary build gets the pure-Go
// fallbacks in transpose_noasm.go and never links a path it cannot exercise.
//
// Removing the tag is exactly the Phase 3 wiring step; the correctness tests
// in transpose_oop_test.go pass in both configurations and only exercise the
// asm in this one, so run them with `-tags fftprobe` when changing it.

package math

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

// transposeAVX2Linked reports whether this build contains the AVX2 transpose
// dispatch at all. It exists so a test can say which of the two files it is
// exercising, rather than reporting CPU support for a path that is not
// compiled in.
const transposeAVX2Linked = true

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
