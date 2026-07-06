//go:build amd64 && asm && !purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

func init() {
	// Override the recursion hooks with AVX2-aware versions.
	recursiveStep64 = mixedRadixRecursivePingPongComplex64AVX2
	recursiveStep128 = mixedRadixRecursivePingPongComplex128AVX2

	// Allow the schedule to emit a composite radix only when the hooks above
	// will actually dispatch it. Schedule-time and dispatch-time decisions
	// share the mixedRadixCodelet64/128 lookup so they cannot drift apart;
	// a mismatch would route a composite radix into the pure Go butterfly,
	// which panics on radices it cannot execute.
	codeletSchedulable64 = func(n int) bool {
		return mixedRadixCodelet64(n) != nil
	}
	codeletSchedulable128 = func(n int) bool {
		return mixedRadixCodelet128(n) != nil
	}
}

// mixedRadixCodelet64 returns the codelet the AVX2 mixed-radix driver will
// dispatch for a size-n sub-transform, or nil if the pure Go recursion must
// handle it. This is the single source of truth for both the scheduler's
// composite-radix predicate and the dispatch hook: an entry qualifies only
// with AVX2 (or better) and both directions available, so a scheduled
// composite radix can always be executed.
func mixedRadixCodelet64(n int) *kernels.CodeletEntry[complex64] {
	entry := kernels.Registry64.Lookup(n, cpu.DetectFeatures())
	if entry == nil || entry.SIMDLevel < kernels.SIMDAVX2 || entry.Forward == nil || entry.Inverse == nil {
		return nil
	}

	return entry
}

// mixedRadixCodelet128 is the complex128 counterpart of mixedRadixCodelet64.
func mixedRadixCodelet128(n int) *kernels.CodeletEntry[complex128] {
	entry := kernels.Registry128.Lookup(n, cpu.DetectFeatures())
	if entry == nil || entry.SIMDLevel < kernels.SIMDAVX2 || entry.Forward == nil || entry.Inverse == nil {
		return nil
	}

	return entry
}

// mixedRadixRecursivePingPongComplex64AVX2 checks for AVX2 codelets before recursing.
func mixedRadixRecursivePingPongComplex64AVX2(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool) {
	// Optimization: dispatch whole sub-transforms to AVX2 codelets when
	// available. Uses the same lookup as the scheduling predicate.
	if n > 1 {
		if entry := mixedRadixCodelet64(n); entry != nil {
			// 1. Prepare Input
			var inputBuf []complex64
			if stride == 1 {
				inputBuf = src[:n]
			} else {
				// Gather strided input into 'work' buffer (scratch space)
				inputBuf = work[:n]
				for i := range n {
					inputBuf[i] = src[i*stride]
				}
			}

			// 2. Prepare Twiddles
			twiddleBuf := make([]complex64, n)
			for i := range n {
				twiddleBuf[i] = twiddle[i*step]
			}

			// 3. Prepare Scratch for Kernel
			kernelScratch := make([]complex64, n)

			// 4. Call Kernel
			codeletTwiddle := twiddleBuf
			if prepared := kernels.GetPreparedTwiddle64(entry, n, inverse); prepared != nil {
				codeletTwiddle = prepared
			}

			if inverse {
				entry.Inverse(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
				// Undo built-in scaling of the Inverse codelet (1/n)
				scale := complex64(complex(float32(n), 0))
				for i := range n {
					dst[i] *= scale
				}
			} else {
				entry.Forward(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
			}

			return
		}
	}

	// Fallback to pure Go implementation.
	mixedRadixRecursivePingPongComplex64(dst, src, work, n, stride, step, radices, twiddle, inverse)
}

// mixedRadixRecursivePingPongComplex128AVX2 is the complex128 version.
func mixedRadixRecursivePingPongComplex128AVX2(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool) {
	if n > 1 {
		if entry := mixedRadixCodelet128(n); entry != nil {
			var inputBuf []complex128
			if stride == 1 {
				inputBuf = src[:n]
			} else {
				inputBuf = work[:n]
				for i := range n {
					inputBuf[i] = src[i*stride]
				}
			}

			twiddleBuf := make([]complex128, n)
			for i := range n {
				twiddleBuf[i] = twiddle[i*step]
			}

			kernelScratch := make([]complex128, n)

			codeletTwiddle := twiddleBuf
			if prepared := kernels.GetPreparedTwiddle128(entry, n, inverse); prepared != nil {
				codeletTwiddle = prepared
			}

			if inverse {
				entry.Inverse(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
				// Undo built-in scaling of the Inverse codelet (1/n)
				scale := complex128(complex(float64(n), 0))
				for i := range n {
					dst[i] *= scale
				}
			} else {
				entry.Forward(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
			}

			return
		}
	}

	mixedRadixRecursivePingPongComplex128(dst, src, work, n, stride, step, radices, twiddle, inverse)
}
