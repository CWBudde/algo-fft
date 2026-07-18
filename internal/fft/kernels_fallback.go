package fft

import (
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// simdTierServesStrategy reports whether a fixed-algorithm SIMD tier (the
// generic SSE/SSE2 radix-2 DIT wrappers, which ignore the strategy argument)
// may serve a plan with the given strategy. Only auto and an explicit
// KernelDIT match the algorithm those wrappers implement; any other forced
// strategy must fall through to the strategy-dispatching auto kernel,
// otherwise the tier silently overrides the caller's algorithm choice (and
// breaks the zero-allocation guarantee of the pure-Go strategies — the SSE
// wrappers recompute bit-reversal tables per call).
func simdTierServesStrategy(strategy KernelStrategy) bool {
	return strategy == KernelAuto || strategy == KernelDIT
}

func fallbackKernel[T Complex](primary, fallback Kernel[T]) Kernel[T] {
	if primary == nil {
		return fallback
	}

	return func(dst, src, twiddle, scratch []T) bool {
		if primary != nil && primary(dst, src, twiddle, scratch) {
			return true
		}

		return fallback(dst, src, twiddle, scratch)
	}
}

func autoKernelComplex64(strategy KernelStrategy) Kernels[complex64] {
	return Kernels[complex64]{
		Forward: func(dst, src, twiddle, scratch []complex64) bool {
			if !m.IsPowerOf2(len(src)) {
				if m.IsMixedRadixSmooth(len(src)) {
					return forwardMixedRadixComplex64(dst, src, twiddle, scratch)
				}

				return false
			}

			switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
			case KernelDIT:
				return forwardDITComplex64(dst, src, twiddle, scratch)
			case KernelStockham:
				return forwardStockhamComplex64(dst, src, twiddle, scratch)
			case KernelSixStep:
				return kernels.ForwardSixStepComplex64(dst, src, twiddle, scratch)
			case KernelEightStep:
				return kernels.ForwardEightStepComplex64(dst, src, twiddle, scratch)
			case KernelSplitRadix:
				return kernels.ForwardSplitRadixComplex64(dst, src, twiddle, scratch)
			default:
				return forwardStockhamComplex64(dst, src, twiddle, scratch)
			}
		},
		Inverse: func(dst, src, twiddle, scratch []complex64) bool {
			if !m.IsPowerOf2(len(src)) {
				if m.IsMixedRadixSmooth(len(src)) {
					return inverseMixedRadixComplex64(dst, src, twiddle, scratch)
				}

				return false
			}

			switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
			case KernelDIT:
				return inverseDITComplex64(dst, src, twiddle, scratch)
			case KernelStockham:
				return inverseStockhamComplex64(dst, src, twiddle, scratch)
			case KernelSixStep:
				return kernels.InverseSixStepComplex64(dst, src, twiddle, scratch)
			case KernelEightStep:
				return kernels.InverseEightStepComplex64(dst, src, twiddle, scratch)
			case KernelSplitRadix:
				return kernels.InverseSplitRadixComplex64(dst, src, twiddle, scratch)
			default:
				return inverseStockhamComplex64(dst, src, twiddle, scratch)
			}
		},
	}
}

func autoKernelComplex128(strategy KernelStrategy) Kernels[complex128] {
	return Kernels[complex128]{
		Forward: func(dst, src, twiddle, scratch []complex128) bool {
			if !m.IsPowerOf2(len(src)) {
				if m.IsMixedRadixSmooth(len(src)) {
					return forwardMixedRadixComplex128(dst, src, twiddle, scratch)
				}

				return false
			}

			switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
			case KernelDIT:
				return forwardDITComplex128(dst, src, twiddle, scratch)
			case KernelStockham:
				return forwardStockhamComplex128(dst, src, twiddle, scratch)
			case KernelSixStep:
				return kernels.ForwardSixStepComplex128(dst, src, twiddle, scratch)
			case KernelEightStep:
				return kernels.ForwardEightStepComplex128(dst, src, twiddle, scratch)
			case KernelSplitRadix:
				return kernels.ForwardSplitRadixComplex128(dst, src, twiddle, scratch)
			default:
				return forwardStockhamComplex128(dst, src, twiddle, scratch)
			}
		},
		Inverse: func(dst, src, twiddle, scratch []complex128) bool {
			if !m.IsPowerOf2(len(src)) {
				if m.IsMixedRadixSmooth(len(src)) {
					return inverseMixedRadixComplex128(dst, src, twiddle, scratch)
				}

				return false
			}

			switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
			case KernelDIT:
				return inverseDITComplex128(dst, src, twiddle, scratch)
			case KernelStockham:
				return inverseStockhamComplex128(dst, src, twiddle, scratch)
			case KernelSixStep:
				return kernels.InverseSixStepComplex128(dst, src, twiddle, scratch)
			case KernelEightStep:
				return kernels.InverseEightStepComplex128(dst, src, twiddle, scratch)
			case KernelSplitRadix:
				return kernels.InverseSplitRadixComplex128(dst, src, twiddle, scratch)
			default:
				return inverseStockhamComplex128(dst, src, twiddle, scratch)
			}
		},
	}
}
