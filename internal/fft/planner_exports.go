package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// Re-export planner types for backward compatibility

// KernelStrategy is a type alias from the planner package.
type KernelStrategy = planner.KernelStrategy

// Kernel strategy constants
const (
	KernelAuto      = planner.KernelAuto
	KernelDIT       = planner.KernelDIT
	KernelStockham  = planner.KernelStockham
	KernelSixStep   = planner.KernelSixStep
	KernelEightStep = planner.KernelEightStep
	KernelBluestein = planner.KernelBluestein
	KernelRecursive = planner.KernelRecursive
)

// Note: CodeletFunc, CodeletEntry, CodeletRegistry, GetRegistry are already defined in kernels.go

// WisdomKey is a type alias from the planner package.
type WisdomKey = planner.WisdomKey

// WisdomEntry is a type alias from the planner package.
type WisdomEntry = planner.WisdomEntry

// Wisdom is a type alias from the planner package.
type Wisdom = planner.Wisdom

// NewWisdom creates a new wisdom cache.
func NewWisdom() *Wisdom {
	return planner.NewWisdom()
}

// DefaultWisdom is the global wisdom cache.
var DefaultWisdom = planner.DefaultWisdom

// Precision constants
const (
	PrecisionComplex64  = planner.PrecisionComplex64
	PrecisionComplex128 = planner.PrecisionComplex128
)

// MakeWisdomKey creates a wisdom key.
func MakeWisdomKey[T Complex](size int, hasSSE2, hasAVX2, hasAVX512, hasNEON bool) WisdomKey {
	return planner.MakeWisdomKey[T](size, hasSSE2, hasAVX2, hasAVX512, hasNEON)
}

// WisdomStore is a type alias from the planner package.
type WisdomStore = planner.WisdomStore

// PlanEstimate is a type alias from the planner package.
type PlanEstimate[T Complex] = planner.PlanEstimate[T]

// EstimatePlan determines the best kernel/codelet for the given size.
func EstimatePlan[T Complex](n int, features cpu.Features, wisdom WisdomStore, forcedStrategy KernelStrategy) PlanEstimate[T] {
	return planner.EstimatePlan[T](n, features, wisdom, forcedStrategy)
}

// HasCodelet returns true if a codelet is available for the given size.
func HasCodelet[T Complex](n int, features cpu.Features) bool {
	return planner.HasCodelet[T](n, features)
}

// CPUFeatureMask returns a bitmask of CPU features.
func CPUFeatureMask(hasSSE2, hasAVX2, hasAVX512, hasNEON bool) uint64 {
	return planner.CPUFeatureMask(hasSSE2, hasAVX2, hasAVX512, hasNEON)
}

// SetKernelStrategy sets the global kernel strategy.
func SetKernelStrategy(strategy KernelStrategy) {
	planner.SetKernelStrategy(strategy)
}

// GetKernelStrategy returns the current global kernel strategy.
func GetKernelStrategy() KernelStrategy {
	return planner.GetKernelStrategy()
}

// RecordBenchmarkDecision records a per-size kernel choice.
func RecordBenchmarkDecision(n int, strategy KernelStrategy) {
	planner.RecordBenchmarkDecision(n, strategy)
}

// ResolveKernelStrategy returns the selected strategy for size n.
func ResolveKernelStrategy(n int) KernelStrategy {
	return planner.ResolveKernelStrategy(n)
}

// ResolveKernelStrategyWithDefault resolves using the provided default.
func ResolveKernelStrategyWithDefault(n int, defaultStrategy KernelStrategy) KernelStrategy {
	return planner.ResolveKernelStrategyWithDefault(n, defaultStrategy)
}

// resolveKernelStrategy is an internal function used by kernel selection.
func resolveKernelStrategy(n int, defaultStrategy KernelStrategy) KernelStrategy {
	return planner.ResolveKernelStrategyWithDefault(n, defaultStrategy)
}

// strategyToAlgorithmName converts a kernel strategy to an algorithm name.
func strategyToAlgorithmName(strategy KernelStrategy) string {
	switch strategy {
	case KernelDIT:
		return "dit_fallback"
	case KernelStockham:
		return "stockham"
	case KernelSixStep:
		return "sixstep"
	case KernelEightStep:
		return "eightstep"
	case KernelBluestein:
		return "bluestein"
	default:
		return "unknown"
	}
}
