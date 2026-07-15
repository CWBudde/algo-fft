package fft

import (
	"strconv"

	"github.com/cwbudde/algo-fft/internal/kernels"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// Re-export kernel types from internal/kernels and internal/planner.
type (
	Kernel[T Complex]          = kernels.Kernel[T]
	Kernels[T Complex]         = kernels.Kernels[T]
	RadixKernel[T Complex]     = kernels.RadixKernel[T]
	CodeletFunc[T Complex]     = planner.CodeletFunc[T]
	CodeletRegistry[T Complex] = planner.CodeletRegistry[T]
	CodeletEntry[T Complex]    = planner.CodeletEntry[T]
	PackedTwiddles[T Complex]  = transform.PackedTwiddles[T]
	SIMDLevel                  = planner.SIMDLevel
)

// Re-export kernel functions.
var (
	// Stockham kernels.
	forwardStockhamComplex64  = kernels.ForwardStockhamComplex64
	inverseStockhamComplex64  = kernels.InverseStockhamComplex64
	forwardStockhamComplex128 = kernels.ForwardStockhamComplex128
	inverseStockhamComplex128 = kernels.InverseStockhamComplex128

	// Packed Stockham kernels.
	StockhamPackedAvailable = transform.StockhamPackedAvailable

	// Registries (direct pointers, not double pointers).
	Registry64  = planner.Registry64
	Registry128 = planner.Registry128
)

// Wrapper functions for generic functions

func ditForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return kernels.DITForward(dst, src, twiddle, scratch)
}

func ditInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return kernels.DITInverse(dst, src, twiddle, scratch)
}

// Precision-specific DIT kernel wrappers.
var (
	forwardDITComplex64  = kernels.ForwardDITComplex64
	inverseDITComplex64  = kernels.InverseDITComplex64
	forwardDITComplex128 = kernels.ForwardDITComplex128
	inverseDITComplex128 = kernels.InverseDITComplex128

	// Size-specific exports for benchmarks/tests.
	ComputeBitReversalIndicesRadix4      = mathpkg.ComputeBitReversalIndicesRadix4
	ComputeBitReversalIndicesRadix4Then2 = mathpkg.ComputeBitReversalIndicesRadix4Then2
	forwardDIT256Complex64               = kernels.ForwardDIT256Complex64
	forwardDIT256Radix4Complex64         = kernels.ForwardDIT256Radix4Complex64

	// Complex128 variants.
)

func sameSlice[T any](a, b []T) bool {
	return kernels.SameSlice(a, b)
}

func stockhamForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return kernels.StockhamForward(dst, src, twiddle, scratch)
}

func stockhamInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return kernels.StockhamInverse(dst, src, twiddle, scratch)
}

func ComputeChirpSequence[T Complex](n int) []T {
	return kernels.ComputeChirpSequence[T](n)
}

// ComputeBluesteinFilter computes the frequency-domain Bluestein filter for
// padded size m. Power-of-two m runs the radix-2 DIT sub-FFT; other 5-smooth
// m (see math.NextHighlyComposite) runs through the mixed-radix engine.
func ComputeBluesteinFilter[T Complex](n, m int, chirp []T, twiddles []T, scratch []T) []T {
	if mathpkg.IsPowerOf2(m) {
		return kernels.ComputeBluesteinFilter[T](n, m, chirp, twiddles, scratch)
	}

	b := kernels.BuildBluesteinSequence(n, m, chirp)
	mustMixedRadix(mixedRadixForward(b, b, twiddles, scratch), m)

	return b
}

// BluesteinConvolution performs the cyclic convolution y = x * b via a padded
// sub-FFT of size m = len(filter). Power-of-two m uses the radix-2 DIT kernels
// with the precomputed bitrev table; other 5-smooth m dispatches to the
// mixed-radix engine (which ignores bitrev).
func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T, bitrev []int) {
	m := len(filter)
	if mathpkg.IsPowerOf2(m) {
		kernels.BluesteinConvolution[T](dst, x, filter, twiddles, scratch, bitrev)
		return
	}

	mustMixedRadix(mixedRadixForward(dst, x, twiddles, scratch), m)

	for i := range dst {
		dst[i] *= filter[i]
	}

	mustMixedRadix(mixedRadixInverse(dst, dst, twiddles, scratch), m)
}

// mustMixedRadix panics when the mixed-radix engine rejects a Bluestein
// sub-FFT size. Plan construction only selects 5-smooth padded sizes, which
// the engine schedules unconditionally, so a failure here is a
// planner/engine contract violation — returning would leave dst partially
// written and surface as a silent wrong answer.
func mustMixedRadix(ok bool, m int) {
	if !ok {
		panic("algofft: mixed-radix engine rejected Bluestein sub-FFT size " + strconv.Itoa(m) +
			" (planner/engine contract violation)")
	}
}

func GetRegistry[T Complex]() *CodeletRegistry[T] {
	return planner.GetRegistry[T]()
}

func butterfly3Forward[T Complex](a0, a1, a2 T) (T, T, T) {
	return kernels.Butterfly3Forward(a0, a1, a2)
}

func butterfly3Inverse[T Complex](a0, a1, a2 T) (T, T, T) {
	return kernels.Butterfly3Inverse(a0, a1, a2)
}

func butterfly4Forward[T Complex](a0, a1, a2, a3 T) (T, T, T, T) {
	return kernels.Butterfly4Forward(a0, a1, a2, a3)
}

func butterfly4Inverse[T Complex](a0, a1, a2, a3 T) (T, T, T, T) {
	return kernels.Butterfly4Inverse(a0, a1, a2, a3)
}

func butterfly5Forward[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	return kernels.Butterfly5Forward(a0, a1, a2, a3, a4)
}

func butterfly5Inverse[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	return kernels.Butterfly5Inverse(a0, a1, a2, a3, a4)
}

func butterfly2[T Complex](a, b, w T) (T, T) {
	return kernels.Butterfly2(a, b, w)
}

// These functions are re-exported in transform_exports.go

// Re-export SIMD level constants.
const (
	SIMDNone   = planner.SIMDNone
	SIMDSSE2   = planner.SIMDSSE2
	SIMDAVX2   = planner.SIMDAVX2
	SIMDAVX512 = planner.SIMDAVX512
	SIMDNEON   = planner.SIMDNEON
)
