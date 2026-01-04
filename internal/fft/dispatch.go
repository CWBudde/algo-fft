package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// Type aliases for planner types used in the fft package.
type (
	KernelStrategy = planner.KernelStrategy
	Wisdom         = planner.Wisdom
	WisdomEntry    = planner.WisdomEntry
	WisdomStore    = planner.WisdomStore
	WisdomKey      = planner.WisdomKey
)

// Generic type alias for PlanEstimate
type PlanEstimate[T Complex] = planner.PlanEstimate[T]

// Re-export precision constants from planner.
const (
	PrecisionComplex64  = planner.PrecisionComplex64
	PrecisionComplex128 = planner.PrecisionComplex128
)

// Re-export kernel strategy constants from planner.
const (
	KernelAuto      = planner.KernelAuto
	KernelDIT       = planner.KernelDIT
	KernelStockham  = planner.KernelStockham
	KernelSixStep   = planner.KernelSixStep
	KernelEightStep = planner.KernelEightStep
	KernelBluestein = planner.KernelBluestein
	KernelRecursive = planner.KernelRecursive
)

// Re-export functions and variables from planner.
var (
	SetKernelStrategy       = planner.SetKernelStrategy
	GetKernelStrategy       = planner.GetKernelStrategy
	RecordBenchmarkDecision = planner.RecordBenchmarkDecision
	ResolveKernelStrategy   = planner.ResolveKernelStrategy
	DefaultWisdom           = planner.DefaultWisdom
	NewWisdom               = planner.NewWisdom
	CPUFeatureMask          = planner.CPUFeatureMask
)

// Wrapper functions for generic planner functions.
func EstimatePlan[T Complex](n int, features cpu.Features, wisdom WisdomStore, strategy KernelStrategy) PlanEstimate[T] {
	return planner.EstimatePlan[T](n, features, wisdom, strategy)
}

func HasCodelet[T Complex](n int, features cpu.Features) bool {
	return planner.HasCodelet[T](n, features)
}

// Kernel and Kernels types are now imported from internal/kernels via kernels.go

// SelectKernels returns the best available kernels for the detected features.
// Currently returns stubs until optimized kernels are implemented.
func SelectKernels[T Complex](features cpu.Features) Kernels[T] {
	var zero T
	switch any(zero).(type) {
	case complex64:
		k := selectKernelsComplex64(features)

		forward, _ := any(k.Forward).(Kernel[T])
		inverse, _ := any(k.Inverse).(Kernel[T])

		return Kernels[T]{
			Forward: forward,
			Inverse: inverse,
		}
	case complex128:
		k := selectKernelsComplex128(features)

		forward, _ := any(k.Forward).(Kernel[T])
		inverse, _ := any(k.Inverse).(Kernel[T])

		return Kernels[T]{
			Forward: forward,
			Inverse: inverse,
		}
	default:
		return Kernels[T]{
			Forward: stubKernel[T],
			Inverse: stubKernel[T],
		}
	}
}

// SelectKernelsWithStrategy returns kernels based on a forced or auto strategy.
func SelectKernelsWithStrategy[T Complex](features cpu.Features, strategy KernelStrategy) Kernels[T] {
	var zero T
	switch any(zero).(type) {
	case complex64:
		k := selectKernelsComplex64WithStrategy(features, strategy)

		forward, _ := any(k.Forward).(Kernel[T])
		inverse, _ := any(k.Inverse).(Kernel[T])

		return Kernels[T]{
			Forward: forward,
			Inverse: inverse,
		}
	case complex128:
		k := selectKernelsComplex128WithStrategy(features, strategy)

		forward, _ := any(k.Forward).(Kernel[T])
		inverse, _ := any(k.Inverse).(Kernel[T])

		return Kernels[T]{
			Forward: forward,
			Inverse: inverse,
		}
	default:
		return Kernels[T]{
			Forward: stubKernel[T],
			Inverse: stubKernel[T],
		}
	}
}

func stubKernel[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}
