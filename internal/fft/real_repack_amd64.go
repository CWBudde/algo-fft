//go:build amd64 && asm && !purego

package fft

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

func inverseRepackComplex64SIMD(dst, src, weight []complex64) int {
	features := cpu.DetectFeatures()
	if features.ForceGeneric {
		return 1
	}

	if features.HasAVX2 {
		return inverseRepackComplex64AVX2(dst, src, weight)
	}
	if features.HasSSE2 {
		return inverseRepackComplex64SSE2(dst, src, weight)
	}
	return 1
}

func inverseRepackComplex64AVX2(dst, src, weight []complex64) int {
	half := len(dst)
	if half < 2 {
		return 1
	}

	limit := half / 2
	if limit >= 1 {
		amd64.InverseRepackComplex64AVX2Asm(dst, src, weight, limit)
	}

	start := limit + 1
	if start < 1 {
		start = 1
	}
	return start
}

func inverseRepackComplex64SSE2(dst, src, weight []complex64) int {
	half := len(dst)
	if half < 2 {
		return 1
	}

	limit := half / 2
	if limit >= 1 {
		amd64.InverseRepackComplex64SSE2Asm(dst, src, weight, limit)
	}

	start := limit + 1
	if start < 1 {
		start = 1
	}
	return start
}

func inverseRepackComplex128SIMD(dst, src, weight []complex128) int {
	return 1
}
