package fft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

// bridgeKernel converts a concretely-typed kernel (returned by the per-arch
// selectKernels* helpers) to the generic kernels.Kernel[T]. The type-switch in
// the callers guarantees T matches the concrete type, so the assertion always
// succeeds; a failure indicates a dispatch bug, so we panic rather than
// silently return a nil kernel. A legitimately nil kernel of the matching
// concrete type still asserts ok == true and is returned as-is.
func bridgeKernel[T Complex](k any) kernels.Kernel[T] {
	kern, ok := k.(kernels.Kernel[T])
	if !ok {
		panic(fmt.Sprintf("algofft: kernel type mismatch, got %T want Kernel[%T]", k, *new(T)))
	}

	return kern
}

// bridgeKernels bridges a matched forward/inverse kernel pair to kernels.Kernels[T].
func bridgeKernels[T Complex](forward, inverse any) kernels.Kernels[T] {
	return kernels.Kernels[T]{
		Forward: bridgeKernel[T](forward),
		Inverse: bridgeKernel[T](inverse),
	}
}

// SelectKernels returns the best available kernels for the detected features.
func SelectKernels[T Complex](features cpu.Features) kernels.Kernels[T] {
	var zero T
	switch any(zero).(type) {
	case complex64:
		k := selectKernelsComplex64(features)
		return bridgeKernels[T](k.Forward, k.Inverse)
	case complex128:
		k := selectKernelsComplex128(features)
		return bridgeKernels[T](k.Forward, k.Inverse)
	default:
		return kernels.Kernels[T]{
			Forward: stubKernel[T],
			Inverse: stubKernel[T],
		}
	}
}

// SelectKernelsWithStrategy returns kernels based on a forced or auto strategy.
func SelectKernelsWithStrategy[T Complex](features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[T] {
	var zero T
	switch any(zero).(type) {
	case complex64:
		k := selectKernelsComplex64WithStrategy(features, strategy)
		return bridgeKernels[T](k.Forward, k.Inverse)
	case complex128:
		k := selectKernelsComplex128WithStrategy(features, strategy)
		return bridgeKernels[T](k.Forward, k.Inverse)
	default:
		return kernels.Kernels[T]{
			Forward: stubKernel[T],
			Inverse: stubKernel[T],
		}
	}
}

func stubKernel[T Complex](dst, src, twiddle, scratch []T) bool {
	return false
}
