//go:build arm64 && asm && !purego

package fft

import (
	arm64 "github.com/cwbudde/algo-fft/internal/asm/arm64"
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

	if features.HasNEON && n >= 2 {
		arm64.ScaleComplex64NEONAsm(dst, scale)
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

	if features.HasNEON && n >= 1 {
		arm64.ScaleComplex128NEONAsm(dst, scale)
		return true
	}

	return false
}
