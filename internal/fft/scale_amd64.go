//go:build amd64 && asm && !purego

package fft

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

func scaleComplex64SIMD(dst []complex64, scale float32) bool {
	features := cpu.DetectFeatures()
	if features.ForceGeneric {
		return false
	}

	n := len(dst)
	if n == 0 {
		return true
	}

	if features.HasAVX2 && n >= 4 {
		amd64.ScaleComplex64AVX2Asm(dst, scale)
		return true
	}

	if features.HasSSE2 {
		amd64.ScaleComplex64SSE2Asm(dst, scale)
		return true
	}

	return false
}

func scaleComplex128SIMD(dst []complex128, scale float64) bool {
	features := cpu.DetectFeatures()
	if features.ForceGeneric {
		return false
	}

	n := len(dst)
	if n == 0 {
		return true
	}

	if features.HasAVX2 && n >= 2 {
		amd64.ScaleComplex128AVX2Asm(dst, scale)
		return true
	}

	if features.HasSSE2 {
		amd64.ScaleComplex128SSE2Asm(dst, scale)
		return true
	}

	return false
}
