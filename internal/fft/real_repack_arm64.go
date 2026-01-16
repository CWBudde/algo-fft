//go:build arm64 && asm && !purego

package fft

import (
	arm64 "github.com/MeKo-Christian/algo-fft/internal/asm/arm64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func inverseRepackComplex64SIMD(dst, src, weight []complex64) int {
	_ = cpu.DetectFeatures()
	_ = arm64.InverseRepackComplex64NEONAsm
	return 1
}

func inverseRepackComplex128SIMD(dst, src, weight []complex128) int {
	return 1
}
