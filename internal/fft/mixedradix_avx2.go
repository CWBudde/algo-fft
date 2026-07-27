//go:build amd64 && !purego

package fft

import (
	"sync"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	"github.com/cwbudde/algo-fft/internal/registry"
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

	// Same lookup once more, this time hoisted out of the recursion: the
	// drivers below dispatch the entry they are handed rather than resolving
	// one per node.
	leafCodelet64 = mixedRadixCodelet64
	leafCodelet128 = mixedRadixCodelet128
}

// mixedRadixCodelet64 returns the codelet the AVX2 mixed-radix driver will
// dispatch for a size-n sub-transform, or nil if the pure Go recursion must
// handle it. This is the single source of truth for both the scheduler's
// composite-radix predicate and the dispatch hook: an entry qualifies only
// with AVX2 (or better) and both directions available, so a scheduled
// composite radix can always be executed.
func mixedRadixCodelet64(n int) *registry.CodeletEntry[complex64] {
	entry := registry.Registry64.Lookup(n, cpu.DetectFeatures())
	if entry == nil || entry.SIMDLevel < fftypes.SIMDAVX2 || entry.Forward == nil || entry.Inverse == nil {
		return nil
	}

	return entry
}

// mixedRadixCodelet128 is the complex128 counterpart of mixedRadixCodelet64.
func mixedRadixCodelet128(n int) *registry.CodeletEntry[complex128] {
	entry := registry.Registry128.Lookup(n, cpu.DetectFeatures())
	if entry == nil || entry.SIMDLevel < fftypes.SIMDAVX2 || entry.Forward == nil || entry.Inverse == nil {
		return nil
	}

	return entry
}

// mixedRadixRecursivePingPongComplex64AVX2 checks for AVX2 codelets before recursing.
func mixedRadixRecursivePingPongComplex64AVX2(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool, leaf *registry.CodeletEntry[complex64]) {
	// Optimization: dispatch whole sub-transforms to AVX2 codelets when
	// available. The entry was resolved once per transform from the schedule's
	// final stage (see leafCodelet64), the only stage the scheduling predicate
	// can match, so testing len(radices) here selects exactly the nodes the
	// per-node registry lookup used to select -- without the lookup.
	if entry := leaf; entry != nil && len(radices) == 1 {
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

		// 2. Prepare Twiddles.
		//
		// Order matters: a codelet that declares a prepared layout ignores
		// the standard table entirely, so gathering one first would be pure
		// waste. Failing that, the cached size-n table is interchangeable
		// with the stride-step gather whenever the recursion invariant
		// n*step == len(twiddle) holds -- which it does at every node the
		// driver reaches. The gather below is the guarded fallback for a
		// caller that passes an oversized table.
		codeletTwiddle := kernels.GetPreparedTwiddle64(entry, n, inverse)
		if codeletTwiddle == nil {
			if leafTwiddleUsable(n, step, len(twiddle)) {
				codeletTwiddle = leafTwiddle64(n)
			} else {
				twPtr := getMRScratch64(n)
				defer mrScratchPool64.Put(twPtr)

				twiddleBuf := *twPtr
				for i := range n {
					twiddleBuf[i] = twiddle[i*step]
				}

				codeletTwiddle = twiddleBuf
			}
		}

		// 3. Prepare Scratch for kernels.Kernel (pooled)
		scrPtr := getMRScratch64(n)
		defer mrScratchPool64.Put(scrPtr)

		kernelScratch := *scrPtr

		// A codelet reports false when it bailed without doing any work;
		// fall through to the pure-Go implementation then.
		if inverse {
			if entry.Inverse(dst[:n], inputBuf, codeletTwiddle, kernelScratch) {
				// Undo built-in scaling of the Inverse codelet (1/n).
				// ScaleComplex64InPlace multiplies by the real factor
				// component-wise and takes the SIMD path when available;
				// `dst[i] *= complex(float32(n), 0)` instead widened every
				// element to complex128 for the same two products (see
				// math.MulComplex64).
				ScaleComplex64InPlace(dst[:n], float32(n))

				return
			}
		} else if entry.Forward(dst[:n], inputBuf, codeletTwiddle, kernelScratch) {
			return
		}
	}

	// Fallback to pure Go implementation.
	mixedRadixRecursivePingPongComplex64(dst, src, work, n, stride, step, radices, twiddle, inverse, leaf)
}

// mixedRadixRecursivePingPongComplex128AVX2 is the complex128 version.
func mixedRadixRecursivePingPongComplex128AVX2(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool, leaf *registry.CodeletEntry[complex128]) {
	// See the complex64 twin for why the leaf entry is passed in.
	if entry := leaf; entry != nil && len(radices) == 1 {
		var inputBuf []complex128
		if stride == 1 {
			inputBuf = src[:n]
		} else {
			inputBuf = work[:n]
			for i := range n {
				inputBuf[i] = src[i*stride]
			}
		}

		// See the complex64 twin for why the prepared-layout check comes
		// first and when the cached size-n table replaces the gather.
		codeletTwiddle := kernels.GetPreparedTwiddle128(entry, n, inverse)
		if codeletTwiddle == nil {
			if leafTwiddleUsable(n, step, len(twiddle)) {
				codeletTwiddle = leafTwiddle128(n)
			} else {
				twPtr := getMRScratch128(n)
				defer mrScratchPool128.Put(twPtr)

				twiddleBuf := *twPtr
				for i := range n {
					twiddleBuf[i] = twiddle[i*step]
				}

				codeletTwiddle = twiddleBuf
			}
		}

		scrPtr := getMRScratch128(n)
		defer mrScratchPool128.Put(scrPtr)

		kernelScratch := *scrPtr

		// A codelet reports false when it bailed without doing any work;
		// fall through to the pure-Go implementation then.
		if inverse {
			if entry.Inverse(dst[:n], inputBuf, codeletTwiddle, kernelScratch) {
				// Undo built-in scaling of the Inverse codelet (1/n); the
				// SIMD-backed helper, as in the complex64 twin.
				ScaleComplex128InPlace(dst[:n], float64(n))

				return
			}
		} else if entry.Forward(dst[:n], inputBuf, codeletTwiddle, kernelScratch) {
			return
		}
	}

	mixedRadixRecursivePingPongComplex128(dst, src, work, n, stride, step, radices, twiddle, inverse, leaf)
}
