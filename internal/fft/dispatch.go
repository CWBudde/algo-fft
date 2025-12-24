package fft

// Kernel reports whether it handled the transform.
// It returns false when no implementation is available.
type Kernel[T Complex] func(dst, src, twiddle, scratch []T, bitrev []int) bool

// Kernels groups forward and inverse kernels for a given precision.
type Kernels[T Complex] struct {
	Forward Kernel[T]
	Inverse Kernel[T]
}

// Features describes CPU capabilities relevant to FFT kernel selection.
// TODO: Populate these from golang.org/x/sys/cpu or custom CPUID logic.
type Features struct {
	HasAVX2       bool
	HasAVX512     bool
	HasSSE2       bool
	HasNEON       bool
	ForceGeneric  bool
	Architecture  string
	Architecture2 string
}

// SelectKernels returns the best available kernels for the detected features.
// Currently returns stubs until optimized kernels are implemented.
func SelectKernels[T Complex](features Features) Kernels[T] {
	var zero T
	switch any(zero).(type) {
	case complex64:
		k := selectKernelsComplex64(features)

		return Kernels[T]{
			Forward: any(k.Forward).(Kernel[T]),
			Inverse: any(k.Inverse).(Kernel[T]),
		}
	case complex128:
		k := selectKernelsComplex128(features)

		return Kernels[T]{
			Forward: any(k.Forward).(Kernel[T]),
			Inverse: any(k.Inverse).(Kernel[T]),
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
