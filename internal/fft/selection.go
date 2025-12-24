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

func selectKernelStrategy(n int) KernelStrategy {
	strategy := GetKernelStrategy()
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
