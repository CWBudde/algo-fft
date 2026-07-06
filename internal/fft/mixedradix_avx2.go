//go:build amd64 && asm && !purego

package fft

import (
	"sync"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

// mrScratchPool64/128 recycle the per-sub-transform gathered-twiddle and kernel
// scratch buffers used when the AVX2 mixed-radix driver dispatches a whole
// sub-transform to a codelet. Each codelet dispatch is a leaf (the function
// returns immediately after, never nesting another dispatch), and the buffers
// are used only for the duration of one synchronous codelet call, so recycling
// them is safe and keeps the AVX2 mixed-radix path allocation-free after
// warm-up. Sizes vary per sub-transform, so the pooled buffers are grown on
// demand and sliced to the requested length.
//
//nolint:gochecknoglobals
var (
	mrScratchPool64  = sync.Pool{New: func() any { s := make([]complex64, 0); return &s }}
	mrScratchPool128 = sync.Pool{New: func() any { s := make([]complex128, 0); return &s }}
)

func getMRScratch64(n int) *[]complex64 {
	p := mrScratchPool64.Get().(*[]complex64) //nolint:forcetypeassert
	if cap(*p) < n {
		*p = make([]complex64, n)
	} else {
		*p = (*p)[:n]
	}

	return p
}

func getMRScratch128(n int) *[]complex128 {
	p := mrScratchPool128.Get().(*[]complex128) //nolint:forcetypeassert
	if cap(*p) < n {
		*p = make([]complex128, n)
	} else {
		*p = (*p)[:n]
	}

	return p
}

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

			// 2. Prepare Twiddles (pooled: reused across sub-transforms)
			twPtr := getMRScratch64(n)
			defer mrScratchPool64.Put(twPtr)

			twiddleBuf := *twPtr
			for i := range n {
				twiddleBuf[i] = twiddle[i*step]
			}

			// 3. Prepare Scratch for Kernel (pooled)
			scrPtr := getMRScratch64(n)
			defer mrScratchPool64.Put(scrPtr)

			kernelScratch := *scrPtr

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

			twPtr := getMRScratch128(n)
			defer mrScratchPool128.Put(twPtr)

			twiddleBuf := *twPtr
			for i := range n {
				twiddleBuf[i] = twiddle[i*step]
			}

			scrPtr := getMRScratch128(n)
			defer mrScratchPool128.Put(scrPtr)

			kernelScratch := *scrPtr

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
