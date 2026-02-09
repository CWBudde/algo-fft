//go:build arm64 && asm && !purego

package fft

import (
	arm64 "github.com/cwbudde/algo-fft/internal/asm/arm64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

// ARM64 SIMD implementations for complex array multiplication.
// Uses NEON assembly when available.

func complexMulArrayComplex64SIMD(dst, a, b []complex64) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	if n < 2 {
		return false
	}

	if features.HasNEON {
		arm64.ComplexMulArrayComplex64NEONAsm(dst, a, b)
		return true
	}

	return false
}

func complexMulArrayComplex128SIMD(dst, a, b []complex128) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	if n < 1 {
		return false
	}

	if features.HasNEON {
		arm64.ComplexMulArrayComplex128NEONAsm(dst, a, b)
		return true
	}

	return false
}

func complexMulArrayInPlaceComplex64SIMD(dst, src []complex64) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	if n < 2 {
		return false
	}

	if features.HasNEON {
		arm64.ComplexMulArrayInPlaceComplex64NEONAsm(dst, src)
		return true
	}

	return false
}

func complexMulArrayInPlaceComplex128SIMD(dst, src []complex128) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	if n < 1 {
		return false
	}

	if features.HasNEON {
		arm64.ComplexMulArrayInPlaceComplex128NEONAsm(dst, src)
		return true
	}

	return false
}
