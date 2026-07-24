package fft

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// simdTierServesStrategy reports whether a fixed-algorithm SIMD tier (the
// generic SSE/SSE2 radix-2 DIT wrappers, which ignore the strategy argument)
// may serve a plan with the given strategy. Only auto and an explicit
// fftypes.KernelDIT match the algorithm those wrappers implement; any other forced
// strategy must fall through to the strategy-dispatching auto kernel,
// otherwise the tier silently overrides the caller's algorithm choice (and
// breaks the zero-allocation guarantee of the pure-Go strategies — the SSE
// wrappers recompute bit-reversal tables per call).
func simdTierServesStrategy(strategy fftypes.KernelStrategy) bool {
	return strategy == fftypes.KernelAuto || strategy == fftypes.KernelDIT
}

func fallbackKernel[T Complex](primary, fallback kernels.Kernel[T]) kernels.Kernel[T] {
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

func autoKernelComplex64(strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	return kernels.Kernels[complex64]{
		Forward: func(dst, src, twiddle, scratch []complex64) bool {
			if !m.IsPowerOf2(len(src)) {
				if m.IsMixedRadixSmooth(len(src)) {
					return forwardMixedRadixComplex64(dst, src, twiddle, scratch)
				}

				return false
			}

			switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
			case fftypes.KernelDIT:
				return kernels.ForwardDITComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelStockham:
				return kernels.ForwardStockhamComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelSixStep:
				return kernels.ForwardSixStepComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelEightStep:
				return kernels.ForwardEightStepComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelSplitRadix:
				return kernels.ForwardSplitRadixComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelFourStep:
				return kernels.ForwardFourStepComplex64(dst, src, twiddle, scratch)
			default:
				return kernels.ForwardStockhamComplex64(dst, src, twiddle, scratch)
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
			case fftypes.KernelDIT:
				return kernels.InverseDITComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelStockham:
				return kernels.InverseStockhamComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelSixStep:
				return kernels.InverseSixStepComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelEightStep:
				return kernels.InverseEightStepComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelSplitRadix:
				return kernels.InverseSplitRadixComplex64(dst, src, twiddle, scratch)
			case fftypes.KernelFourStep:
				return kernels.InverseFourStepComplex64(dst, src, twiddle, scratch)
			default:
				return kernels.InverseStockhamComplex64(dst, src, twiddle, scratch)
			}
		},
	}
}

func autoKernelComplex128(strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	return kernels.Kernels[complex128]{
		Forward: func(dst, src, twiddle, scratch []complex128) bool {
			if !m.IsPowerOf2(len(src)) {
				if m.IsMixedRadixSmooth(len(src)) {
					return forwardMixedRadixComplex128(dst, src, twiddle, scratch)
				}

				return false
			}

			switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
			case fftypes.KernelDIT:
				return kernels.ForwardDITComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelStockham:
				return kernels.ForwardStockhamComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelSixStep:
				return kernels.ForwardSixStepComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelEightStep:
				return kernels.ForwardEightStepComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelSplitRadix:
				return kernels.ForwardSplitRadixComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelFourStep:
				return kernels.ForwardFourStepComplex128(dst, src, twiddle, scratch)
			default:
				return kernels.ForwardStockhamComplex128(dst, src, twiddle, scratch)
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
			case fftypes.KernelDIT:
				return kernels.InverseDITComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelStockham:
				return kernels.InverseStockhamComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelSixStep:
				return kernels.InverseSixStepComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelEightStep:
				return kernels.InverseEightStepComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelSplitRadix:
				return kernels.InverseSplitRadixComplex128(dst, src, twiddle, scratch)
			case fftypes.KernelFourStep:
				return kernels.InverseFourStepComplex128(dst, src, twiddle, scratch)
			default:
				return kernels.InverseStockhamComplex128(dst, src, twiddle, scratch)
			}
		},
	}
}
