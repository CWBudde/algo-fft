package kernels

import (
	"math/bits"

	"github.com/cwbudde/algo-fft/internal/fftypes"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// Complex is a type alias for the complex number constraint.
type Complex = fftypes.Complex

// KernelStrategy is a type alias for the kernel strategy enum.
type KernelStrategy = fftypes.KernelStrategy

// Strategy constants.
const (
	KernelAuto      = fftypes.KernelAuto
	KernelDIT       = fftypes.KernelDIT
	KernelStockham  = fftypes.KernelStockham
	KernelBluestein = fftypes.KernelBluestein
)

// Re-export math utilities used by kernels.
var (
	IsPowerOf2 = mathpkg.IsPowerOf2
	isPowerOf3 = mathpkg.IsPowerOf3
	isPowerOf4 = mathpkg.IsPowerOf4
	isPowerOf5 = mathpkg.IsPowerOf5
)

// log2 returns the base-2 logarithm of n using bits.Len() for efficiency.
func log2(n int) int {
	return bits.Len(uint(n)) - 1
}

// ComputeTwiddleFactors is a wrapper for the generic math function.
func ComputeTwiddleFactors[T Complex](n int) []T {
	return mathpkg.ComputeTwiddleFactors[T](n)
}

// Kernel reports whether it handled the transform.
// It returns false when no implementation is available.
type Kernel[T Complex] func(dst, src, twiddle, scratch []T) bool

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
