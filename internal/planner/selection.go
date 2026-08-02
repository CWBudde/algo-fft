package planner

import (
	"math"
)

const ditAutoThreshold = 1024

// ResolveKernelStrategy returns the heuristically-selected strategy for size n.
// Selection order for KernelAuto:
//  1. Non-power-of-two lengths the mixed-radix engine executes: mixed-radix
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

	// Non-power-of-two lengths the mixed-radix engine executes take that
	// engine unconditionally: the kernel dispatch (internal/fft's
	// autoKernelComplex64/128) checks the length before it looks at the
	// strategy, so no power-of-two strategy — forced or heuristic — ever runs
	// here. Reporting one would name a route that never executes. The
	// exception is an explicitly forced Bluestein, which the plan layer
	// honors and really does run, and a forced Recursive, which plan
	// construction rejects for these lengths rather than silently rerouting.
	if !IsPowerOf2(n) && MixedRadixEligible(n) &&
		strategy != KernelBluestein && strategy != KernelRecursive {
		return KernelMixedRadix
	}

	if strategy != KernelAuto {
		if !isSquareSize(n) && strategy == KernelSixStep {
			return fallbackKernelStrategy(n)
		}

		if strategy == KernelFourStep && (n < 4 || !IsPowerOf2(n)) {
			return fallbackKernelStrategy(n)
		}

		if strategy == KernelMixedRadix {
			// A forced mixed-radix at a length the engine is not the route
			// for (power of two, or Bluestein-bound) would be as dishonest in
			// the other direction.
			return fallbackKernelStrategy(n)
		}

		return strategy
	}

	// Power-of-two squares are deliberately NOT special-cased: measured over
	// every candidate strategy at the only sizes the square branch can reach
	// (2^18, 2^20, and 2^22), the plain size heuristic below — Stockham — wins
	// or ties against six-step, four-step and split-radix on both
	// the SIMD and purego builds, at both precisions and in both directions
	// (BenchmarkSquareAutoRule, i7-1255U/AVX2).
	//
	// This replaces two earlier rules that measured as losses:
	//   - split-radix for [2^18, 2^22): Stockham beats it in every arm except
	//     purego 2^18 complex64 forward, where it trails by 3% (noise). At the
	//     other extreme split-radix costs 2x — 2^20 complex128 forward is
	//     80.3 ms vs six-step's 39.3 ms and Stockham's 49.7 ms.
	//   - six-step for powers of two >= 2^22: at 2^22 complex64 Stockham runs
	//     157/171 ms (fwd/inv) against six-step's 201/269 ms on the SIMD
	//     build, and 102/113 vs 203/247 ms on purego. (These numbers were
	//     originally attributed to the now-removed KernelEightStep, which was
	//     a duplicate of six-step's implementation and always measured
	//     identically to it.)
	//
	// One arm dissents and is accepted knowingly: 2^20 complex128 forward
	// prefers six-step (39.3 ms) over Stockham (49.7 ms) on the SIMD build.
	// A precision- and direction-blind rule cannot capture that, the same
	// size's other three arms all favor Stockham, and Stockham still beats the
	// split-radix it replaces there by 1.6x. Wisdom/measure modes pick per
	// machine where it matters (see selectStrategiesToTest in internal/fft).
	//
	// A non-power-of-two square rule used to live here, returning six-step
	// above 2^18 on the grounds that "it executes through the mixed-radix
	// engine". It did — which is exactly why the rule was a label and never a
	// route: the six-step kernel was never called for those lengths. Every
	// square it could reach is either mixed-radix executable (returned above)
	// or Bluestein-bound (EstimatePlan answers before asking here), so the rule
	// was unreachable in effect and is gone.

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
