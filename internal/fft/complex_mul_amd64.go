//go:build amd64 && asm && !purego

package fft

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

// AMD64 SIMD implementations for complex array multiplication.
// Uses AVX2 assembly when available, falls back to generic.

func complexMulArrayComplex64SIMD(dst, a, b []complex64) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	// Need at least 4 elements for AVX2 (processes 4 complex64 at a time)
	if n < 4 {
		return false
	}

	if features.HasAVX2 {
		complexMulArrayComplex64AVX2(dst, a, b)
		return true
	}

	return false
}

func complexMulArrayComplex128SIMD(dst, a, b []complex128) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	// Need at least 2 elements for AVX2 (processes 2 complex128 at a time)
	if n < 2 {
		return false
	}

	if features.HasAVX2 {
		complexMulArrayComplex128AVX2(dst, a, b)
		return true
	}

	return false
}

func complexMulArrayInPlaceComplex64SIMD(dst, src []complex64) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	// Need at least 4 elements for AVX2 (processes 4 complex64 at a time)
	if n < 4 {
		return false
	}

	if features.HasAVX2 {
		complexMulArrayInPlaceComplex64AVX2(dst, src)
		return true
	}

	return false
}

func complexMulArrayInPlaceComplex128SIMD(dst, src []complex128) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	// Need at least 2 elements for AVX2 (processes 2 complex128 at a time)
	if n < 2 {
		return false
	}

	if features.HasAVX2 {
		complexMulArrayInPlaceComplex128AVX2(dst, src)
		return true
	}

	return false
}

// AVX2 implementations - call assembly routines.

func complexMulArrayComplex64AVX2(dst, a, b []complex64) {
	amd64.ComplexMulArrayComplex64AVX2Asm(dst, a, b)
}

func complexMulArrayComplex128AVX2(dst, a, b []complex128) {
	amd64.ComplexMulArrayComplex128AVX2Asm(dst, a, b)
}

func complexMulArrayInPlaceComplex64AVX2(dst, src []complex64) {
	amd64.ComplexMulArrayInPlaceComplex64AVX2Asm(dst, src)
}

func complexMulArrayInPlaceComplex128AVX2(dst, src []complex128) {
	amd64.ComplexMulArrayInPlaceComplex128AVX2Asm(dst, src)
}
