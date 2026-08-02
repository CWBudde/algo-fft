//go:build arm64 && !purego

package fft

import (
	arm64 "github.com/cwbudde/algo-fft/internal/asm/arm64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

func inverseRepackComplex64SIMD(dst, src, weight []complex64) int {
	features := cpu.DetectFeatures()
	if features.ForceGeneric || !features.HasNEON {
		return 1
	}

	half := len(dst)

	// The vector loop consumes blocks of 2 pair-bins (k and its mirror
	// half-k); it must stop before k meets its mirror, so only full blocks
	// within k <= (half-1)/2 are handled. See the amd64 complex128 twin
	// (inverseRepackComplex128SIMD in real_repack_amd64.go) for the same
	// shape.
	count := (half - 1) / 2 / 2 * 2
	if count < 2 {
		return 1
	}

	arm64.InverseRepackComplex64NEONAsm(dst, src, weight, count)

	return count + 1
}

func inverseRepackComplex128SIMD(dst, src, weight []complex128) int {
	features := cpu.DetectFeatures()
	if features.ForceGeneric || !features.HasNEON {
		return 1
	}

	half := len(dst)

	// Same block contract as the complex64 path above.
	count := (half - 1) / 2 / 2 * 2
	if count < 2 {
		return 1
	}

	arm64.InverseRepackComplex128NEONAsm(dst, src, weight, count)

	return count + 1
}
