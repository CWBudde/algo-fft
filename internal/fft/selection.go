package fft

import (
	"sync"
	"sync/atomic"
)

// KernelStrategy controls how plans choose between DIT and Stockham kernels.
type KernelStrategy uint32

const (
	KernelAuto KernelStrategy = iota
	KernelDIT
	KernelStockham
)

var kernelStrategy uint32 = uint32(KernelAuto)

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

	if strategy != KernelDIT && strategy != KernelStockham {
		return
	}

	benchMu.Lock()

	benchDecisions[n] = strategy

	benchMu.Unlock()
}

// ResolveKernelStrategy returns the selected strategy for size n using
// the global override, benchmark cache, and size threshold.
func ResolveKernelStrategy(n int) KernelStrategy {
	return resolveKernelStrategy(n, KernelAuto)
}

func resolveKernelStrategy(n int, defaultStrategy KernelStrategy) KernelStrategy {
	strategy := defaultStrategy
	if strategy == KernelAuto {
		strategy = GetKernelStrategy()
	}

	if strategy != KernelAuto {
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

	if n <= ditAutoThreshold {
		return KernelDIT
	}

	return KernelStockham
}
