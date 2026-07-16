//go:build amd64 && !purego

package fft

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

// recombineForwardComplex64SIMD processes bins k = 1..count with AVX2 and
// returns the first unprocessed index (1 when nothing was handled).
func recombineForwardComplex64SIMD(dst, src, weight []complex64) int {
	features := cpu.DetectFeatures()
	if features.ForceGeneric {
		return 1
	}

	half := len(src)

	if features.HasAVX2 {
		// The vector loop consumes full blocks of 4 bins starting at k=1; the
		// reversed load for a block ending at k+3 <= half-1 stays in bounds.
		count := (half - 1) / 4 * 4
		if count < 4 {
			return 1
		}

		amd64.RecombineForwardComplex64AVX2Asm(dst, src, weight, count)

		return count + 1
	}

	if features.HasSSE3 {
		// Full blocks of 2 bins starting at k=1.
		count := (half - 1) / 2 * 2
		if count < 2 {
			return 1
		}

		amd64.RecombineForwardComplex64SSE3Asm(dst, src, weight, count)

		return count + 1
	}

	return 1
}

func recombineForwardComplex128SIMD(dst, src, weight []complex128) int {
	features := cpu.DetectFeatures()
	if features.ForceGeneric {
		return 1
	}

	half := len(src)

	if features.HasAVX2 {
		// Full blocks of 2 bins starting at k=1.
		count := (half - 1) / 2 * 2
		if count < 2 {
			return 1
		}

		amd64.RecombineForwardComplex128AVX2Asm(dst, src, weight, count)

		return count + 1
	}

	if features.HasSSE3 {
		// One bin per iteration, so the SSE3 kernel covers the whole range.
		count := half - 1
		if count < 1 {
			return 1
		}

		amd64.RecombineForwardComplex128SSE3Asm(dst, src, weight, count)

		return count + 1
	}

	return 1
}
