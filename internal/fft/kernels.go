package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/kernels"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
	"github.com/MeKo-Christian/algo-fft/internal/transform"
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
	forwardDIT4Radix4Complex64           = kernels.ForwardDIT4Radix4Complex64
	inverseDIT4Radix4Complex64           = kernels.InverseDIT4Radix4Complex64
	forwardDIT8Radix2Complex64           = kernels.ForwardDIT8Radix2Complex64
	inverseDIT8Radix2Complex64           = kernels.InverseDIT8Radix2Complex64
	forwardDIT8Radix4Complex64           = kernels.ForwardDIT8Radix4Complex64
	inverseDIT8Radix4Complex64           = kernels.InverseDIT8Radix4Complex64
	forwardDIT16Complex64                = kernels.ForwardDIT16Complex64
	inverseDIT16Complex64                = kernels.InverseDIT16Complex64
	forwardDIT16Radix4Complex64          = kernels.ForwardDIT16Radix4Complex64
	inverseDIT16Radix4Complex64          = kernels.InverseDIT16Radix4Complex64
	forwardDIT32Complex64                = kernels.ForwardDIT32Complex64
	inverseDIT32Complex64                = kernels.InverseDIT32Complex64
	forwardDIT64Complex64                = kernels.ForwardDIT64Complex64
	inverseDIT64Complex64                = kernels.InverseDIT64Complex64
	forwardDIT64Radix4Complex64          = kernels.ForwardDIT64Radix4Complex64
	inverseDIT64Radix4Complex64          = kernels.InverseDIT64Radix4Complex64
	forwardDIT128Complex64               = kernels.ForwardDIT128Complex64
	inverseDIT128Complex64               = kernels.InverseDIT128Complex64
	forwardDIT256Complex64               = kernels.ForwardDIT256Complex64
	inverseDIT256Complex64               = kernels.InverseDIT256Complex64
	forwardDIT256Radix4Complex64         = kernels.ForwardDIT256Radix4Complex64
	inverseDIT256Radix4Complex64         = kernels.InverseDIT256Radix4Complex64
	forwardDIT512Complex64               = kernels.ForwardDIT512Complex64
	inverseDIT512Complex64               = kernels.InverseDIT512Complex64
	forwardDIT512Radix4Then2Complex64    = kernels.ForwardDIT512Radix4Then2Complex64
	inverseDIT512Radix4Then2Complex64    = kernels.InverseDIT512Radix4Then2Complex64

	// Complex128 variants.
	forwardDIT4Radix4Complex128        = kernels.ForwardDIT4Radix4Complex128
	inverseDIT4Radix4Complex128        = kernels.InverseDIT4Radix4Complex128
	forwardDIT8Radix2Complex128        = kernels.ForwardDIT8Radix2Complex128
	inverseDIT8Radix2Complex128        = kernels.InverseDIT8Radix2Complex128
	forwardDIT8Radix4Complex128        = kernels.ForwardDIT8Radix4Complex128
	inverseDIT8Radix4Complex128        = kernels.InverseDIT8Radix4Complex128
	forwardDIT16Complex128             = kernels.ForwardDIT16Complex128
	inverseDIT16Complex128             = kernels.InverseDIT16Complex128
	forwardDIT16Radix4Complex128       = kernels.ForwardDIT16Radix4Complex128
	inverseDIT16Radix4Complex128       = kernels.InverseDIT16Radix4Complex128
	forwardDIT32Complex128             = kernels.ForwardDIT32Complex128
	inverseDIT32Complex128             = kernels.InverseDIT32Complex128
	forwardDIT64Complex128             = kernels.ForwardDIT64Complex128
	inverseDIT64Complex128             = kernels.InverseDIT64Complex128
	forwardDIT64Radix4Complex128       = kernels.ForwardDIT64Radix4Complex128
	inverseDIT64Radix4Complex128       = kernels.InverseDIT64Radix4Complex128
	forwardDIT128Complex128            = kernels.ForwardDIT128Complex128
	inverseDIT128Complex128            = kernels.InverseDIT128Complex128
	forwardDIT256Complex128            = kernels.ForwardDIT256Complex128
	inverseDIT256Complex128            = kernels.InverseDIT256Complex128
	forwardDIT256Radix4Complex128      = kernels.ForwardDIT256Radix4Complex128
	inverseDIT256Radix4Complex128      = kernels.InverseDIT256Radix4Complex128
	forwardDIT512Complex128            = kernels.ForwardDIT512Complex128
	inverseDIT512Complex128            = kernels.InverseDIT512Complex128
	forwardDIT512Radix4Then2Complex128 = kernels.ForwardDIT512Radix4Then2Complex128
	inverseDIT512Radix4Then2Complex128 = kernels.InverseDIT512Radix4Then2Complex128
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

func ComputeBluesteinFilter[T Complex](n, m int, chirp []T, twiddles []T, scratch []T) []T {
	return kernels.ComputeBluesteinFilter[T](n, m, chirp, twiddles, scratch)
}

func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T) {
	kernels.BluesteinConvolution[T](dst, x, filter, twiddles, scratch)
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
