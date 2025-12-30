package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

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
