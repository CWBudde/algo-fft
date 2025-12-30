package kernels

import (
	"github.com/MeKo-Christian/algo-fft/internal/fftypes"
	"github.com/MeKo-Christian/algo-fft/internal/math"
)

// Complex is a type alias for the complex number constraint.
type Complex = fftypes.Complex

// KernelStrategy is a type alias for the kernel strategy enum.
type KernelStrategy = fftypes.KernelStrategy

// Strategy constants
const (
	KernelAuto      = fftypes.KernelAuto
	KernelDIT       = fftypes.KernelDIT
	KernelStockham  = fftypes.KernelStockham
	KernelBluestein = fftypes.KernelBluestein
)

// Re-export math utilities used by kernels
var (
	ComputeBitReversalIndices = math.ComputeBitReversalIndices
	IsPowerOf2                = math.IsPowerOf2
	isPowerOf3                = math.IsPowerOf3
	isPowerOf4                = math.IsPowerOf4
	isPowerOf5                = math.IsPowerOf5
	log2                      = math.Log2
)

// Kernel reports whether it handled the transform.
// It returns false when no implementation is available.
type Kernel[T Complex] func(dst, src, twiddle, scratch []T, bitrev []int) bool

// Kernels groups forward and inverse kernels for a given precision.
type Kernels[T Complex] struct {
	Forward Kernel[T]
	Inverse Kernel[T]
}

// RadixKernel describes a pluggable radix implementation for mixed-radix FFTs.
// Implementations should return false when they do not support the given length.
type RadixKernel[T Complex] struct {
	Name    string
	Radix   int
	Forward Kernel[T]
	Inverse Kernel[T]
}
