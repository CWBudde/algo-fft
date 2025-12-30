package fft

import (
	"sync"
	"sync/atomic"
)

// KernelStrategy controls how plans choose between DIT, Stockham, and step kernels.
type KernelStrategy uint32

const (
	KernelAuto KernelStrategy = iota
	KernelDIT
	KernelStockham
	KernelSixStep
	KernelEightStep
	KernelBluestein
	KernelRecursive // Recursive decomposition with codelet leaves
)

//nolint:gochecknoglobals
var kernelStrategy uint32 = uint32(KernelAuto)

//nolint:gochecknoglobals
var (
	benchMu        sync.RWMutex
	benchDecisions = make(map[int]KernelStrategy)
)

// SetKernelStrategy overrides the global kernel selection strategy.
// Use KernelAuto to restore automatic selection.
func SetKernelStrategy(strategy KernelStrategy) {
	atomic.StoreUint32(&kernelStrategy, uint32(strategy))
}

// GetKernelStrategy returns the current global kernel selection strategy.
func GetKernelStrategy() KernelStrategy {
	return KernelStrategy(atomic.LoadUint32(&kernelStrategy))
}

// RecordBenchmarkDecision stores a per-size kernel choice.
// This is used only when KernelAuto is active.
func RecordBenchmarkDecision(n int, strategy KernelStrategy) {
	if n <= 0 {
		return
	}

	switch strategy {
	case KernelDIT, KernelStockham, KernelSixStep, KernelEightStep:
	default:
		return
	}

	benchMu.Lock()

	benchDecisions[n] = strategy

	benchMu.Unlock()
}

// ResolveKernelStrategy returns the selected strategy for size n.
// Selection order for KernelAuto:
//  1. Global override (SetKernelStrategy)
//  2. Per-size benchmark decision cache
//  3. Square-size transforms: prefer six/eight-step for large sizes
//  4. Size threshold: DIT for <= ditAutoThreshold, Stockham otherwise
func ResolveKernelStrategy(n int) KernelStrategy {
	return resolveKernelStrategy(n, KernelAuto)
}

// ResolveKernelStrategyWithDefault resolves using the provided default when auto is selected.
func ResolveKernelStrategyWithDefault(n int, defaultStrategy KernelStrategy) KernelStrategy {
	return resolveKernelStrategy(n, defaultStrategy)
}

func resolveKernelStrategy(n int, defaultStrategy KernelStrategy) KernelStrategy {
	strategy := defaultStrategy
	if strategy == KernelAuto {
		strategy = GetKernelStrategy()
	}

	if strategy != KernelAuto {
		if !isSquareSize(n) && (strategy == KernelSixStep || strategy == KernelEightStep) {
			return fallbackKernelStrategy(n)
		}

		return strategy
	}

	if n > 0 {
		benchMu.RLock()

		decision, ok := benchDecisions[n]

		benchMu.RUnlock()

		if ok {
			return decision
		}
	}

	m := intSqrt(n)
	if m*m == n {
		if n >= 1<<22 {
			return KernelEightStep
		}

		if n >= 1<<18 {
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
