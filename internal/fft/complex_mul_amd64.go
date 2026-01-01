//go:build amd64

package fft

import "github.com/MeKo-Christian/algo-fft/internal/cpu"

// AMD64 SIMD implementations for complex array multiplication.
// Uses AVX2 when available, falls back to SSE2, then generic.

func complexMulArrayComplex64SIMD(dst, a, b []complex64) bool {
	features := cpu.DetectFeatures()
	n := len(dst)

	// Need at least 4 elements for SSE2 (processes 2 complex64 = 4 floats at a time)
	if n < 2 {
		return false
	}

	if features.HasAVX2 && n >= 4 {
		complexMulArrayComplex64AVX2(dst, a, b)
		return true
	}

	if features.HasSSE2 {
		complexMulArrayComplex64SSE2(dst, a, b)
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

	if features.HasAVX2 && n >= 2 {
		complexMulArrayComplex128AVX2(dst, a, b)
		return true
	}

	if features.HasSSE2 {
		complexMulArrayComplex128SSE2(dst, a, b)
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

	if features.HasAVX2 && n >= 4 {
		complexMulArrayInPlaceComplex64AVX2(dst, src)
		return true
	}

	if features.HasSSE2 {
		complexMulArrayInPlaceComplex64SSE2(dst, src)
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

	if features.HasAVX2 && n >= 2 {
		complexMulArrayInPlaceComplex128AVX2(dst, src)
		return true
	}

	if features.HasSSE2 {
		complexMulArrayInPlaceComplex128SSE2(dst, src)
		return true
	}

	return false
}

// SSE2 implementations (pure Go for now, can be replaced with assembly later).

func complexMulArrayComplex64SSE2(dst, a, b []complex64) {
	// Process pairs for potential SIMD, with scalar cleanup
	n := len(dst)
	i := 0

	// Main loop: process 2 elements at a time
	for ; i+1 < n; i += 2 {
		dst[i] = a[i] * b[i]
		dst[i+1] = a[i+1] * b[i+1]
	}

	// Cleanup: remaining element
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayComplex128SSE2(dst, a, b []complex128) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayInPlaceComplex64SSE2(dst, src []complex64) {
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

func complexMulArrayInPlaceComplex128SSE2(dst, src []complex128) {
	for i := range dst {
		dst[i] *= src[i]
	}
}

// AVX2 implementations (pure Go for now, can be replaced with assembly later).

func complexMulArrayComplex64AVX2(dst, a, b []complex64) {
	n := len(dst)
	i := 0

	// Main loop: process 4 elements at a time (8 floats = 256 bits)
	for ; i+3 < n; i += 4 {
		dst[i] = a[i] * b[i]
		dst[i+1] = a[i+1] * b[i+1]
		dst[i+2] = a[i+2] * b[i+2]
		dst[i+3] = a[i+3] * b[i+3]
	}

	// Cleanup
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayComplex128AVX2(dst, a, b []complex128) {
	n := len(dst)
	i := 0

	// Main loop: process 2 elements at a time (4 doubles = 256 bits)
	for ; i+1 < n; i += 2 {
		dst[i] = a[i] * b[i]
		dst[i+1] = a[i+1] * b[i+1]
	}

	// Cleanup
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayInPlaceComplex64AVX2(dst, src []complex64) {
	n := len(dst)
	i := 0

	for ; i+3 < n; i += 4 {
		dst[i] *= src[i]
		dst[i+1] *= src[i+1]
		dst[i+2] *= src[i+2]
		dst[i+3] *= src[i+3]
	}

	for ; i < n; i++ {
		dst[i] *= src[i]
	}
}

func complexMulArrayInPlaceComplex128AVX2(dst, src []complex128) {
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
