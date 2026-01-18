//go:build arm64 && asm && !purego

package fft

import (
	arm64 "github.com/MeKo-Christian/algo-fft/internal/asm/arm64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func inverseRepackComplex64SIMD(dst, src, weight []complex64) int {
	features := cpu.DetectFeatures()
	if features.ForceGeneric || !features.HasNEON {
		return 1
	}

	half := len(dst)
	if half < 2 {
		return 1
	}

	limit := half / 2
	if limit >= 1 {
		arm64.InverseRepackComplex64NEONAsm(dst, src, weight, limit)
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
