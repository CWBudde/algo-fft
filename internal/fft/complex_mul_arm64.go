//go:build arm64 && (!asm || purego)

package fft

import "github.com/MeKo-Christian/algo-fft/internal/cpu"

// ARM64 SIMD implementations for complex array multiplication.
// Uses NEON when available.

func complexMulArrayComplex64SIMD(dst, a, b []complex64) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	// Need at least 2 elements for NEON (processes 2 complex64 = 4 floats at a time)
	if n < 2 {
		return false
	}

	if features.HasNEON {
		complexMulArrayComplex64NEON(dst, a, b)
		return true
	}

	return false
}

func complexMulArrayComplex128SIMD(dst, a, b []complex128) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	if n < 2 {
		return false
	}

	if features.HasNEON {
		complexMulArrayComplex128NEON(dst, a, b)
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
		complexMulArrayInPlaceComplex64NEON(dst, src)
		return true
	}

	return false
}

func complexMulArrayInPlaceComplex128SIMD(dst, src []complex128) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	if n < 2 {
		return false
	}

	if features.HasNEON {
		complexMulArrayInPlaceComplex128NEON(dst, src)
		return true
	}

	return false
}

// NEON implementations (pure Go fallback).

func complexMulArrayComplex64NEON(dst, a, b []complex64) {
	n := len(dst)
	i := 0

	// Main loop: process 2 elements at a time (4 floats = 128 bits)
	for ; i+1 < n; i += 2 {
		dst[i] = a[i] * b[i]
		dst[i+1] = a[i+1] * b[i+1]
	}

	// Cleanup
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayComplex128NEON(dst, a, b []complex128) {
	// NEON can process 2x64-bit floats at a time
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayInPlaceComplex64NEON(dst, src []complex64) {
	n := len(dst)
	i := 0

	for ; i+1 < n; i += 2 {
		dst[i] *= src[i]
		dst[i+1] *= src[i+1]
	}

	for ; i < n; i++ {
		dst[i] *= src[i]
	}
}

func complexMulArrayInPlaceComplex128NEON(dst, src []complex128) {
	for i := range dst {
		dst[i] *= src[i]
	}
}
