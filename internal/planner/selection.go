package planner

import (
	"math"
)

const ditAutoThreshold = 1024

// ResolveKernelStrategy returns the heuristically-selected strategy for size n.
// Selection order for KernelAuto:
//  1. Square-size transforms: prefer six/eight-step for large sizes
//  2. Size threshold: DIT for <= ditAutoThreshold, Stockham otherwise
//
// Per-plan overrides are supplied via PlanOptions.Strategy (threaded through as the
// default in ResolveKernelStrategyWithDefault); tuning decisions are persisted per
// instance via the Wisdom cache. There is no process-global strategy state.
func ResolveKernelStrategy(n int) KernelStrategy {
	return resolveKernelStrategy(n, KernelAuto)
}

// ResolveKernelStrategyWithDefault resolves using the provided default when auto is selected.
func ResolveKernelStrategyWithDefault(n int, defaultStrategy KernelStrategy) KernelStrategy {
	return resolveKernelStrategy(n, defaultStrategy)
}

func resolveKernelStrategy(n int, defaultStrategy KernelStrategy) KernelStrategy {
	strategy := defaultStrategy

	if strategy != KernelAuto {
		if !isSquareSize(n) && (strategy == KernelSixStep || strategy == KernelEightStep) {
			return fallbackKernelStrategy(n)
		}

		return strategy
	}

	m := intSqrt(n)
	if m*m == n {
		if n >= 1<<22 {
			return KernelEightStep
		}

		if n >= 1<<18 {
			// Power-of-two squares in [2^18, 2^22) — 512^2 and 1024^2 —
			// previously resolved to six-step, whose scalar transpose
			// dominates at these sizes (the SIMD transpose kernels stop at
			// 128x128). Split-radix measured ~2x faster for both directions,
			// both precisions, on both the SIMD and purego builds
			// (BenchmarkSplitRadixVsIncumbents). Re-measured after the P4.3
			// cache-blocked transpose landed: six-step gained 10-17% at
			// these sizes but split-radix still wins 1.2-1.6x, so the rule
			// stands. Non-power-of-two squares keep six-step: they execute
			// through the mixed-radix engine anyway, and split-radix would
			// decline them.
			if IsPowerOf2(n) {
				return KernelSplitRadix
			}

			return KernelSixStep
		}
	}

	if n <= ditAutoThreshold {
		return KernelDIT
	}

	return KernelStockham
}

func isSquareSize(n int) bool {
	if n <= 0 {
		return false
	}

	root := intSqrt(n)

	return root*root == n
}

func fallbackKernelStrategy(n int) KernelStrategy {
	if n <= ditAutoThreshold {
		return KernelDIT
	}

	return KernelStockham
}

// intSqrt computes the integer square root of n.
func intSqrt(n int) int {
	if n <= 0 {
		return 0
	}

	root := int(math.Sqrt(float64(n)))

	// Handle rounding errors
	if root*root > n {
		root--
	} else if (root+1)*(root+1) <= n {
		root++
	}

	return root
}
