package fft

import "github.com/MeKo-Christian/algo-fft/internal/kernels"

// Re-export kernel types from internal/kernels
type (
	Kernel[T Complex]          = kernels.Kernel[T]
	Kernels[T Complex]         = kernels.Kernels[T]
	RadixKernel[T Complex]     = kernels.RadixKernel[T]
	CodeletFunc[T Complex]     = kernels.CodeletFunc[T]
	CodeletRegistry[T Complex] = kernels.CodeletRegistry[T]
	CodeletEntry[T Complex]    = kernels.CodeletEntry[T]
	PackedTwiddles[T Complex]  = kernels.PackedTwiddles[T]
	BitrevFunc                 = kernels.BitrevFunc
	SIMDLevel                  = kernels.SIMDLevel
)

// Re-export kernel functions
var (
	// Stockham kernels
	forwardStockhamComplex64  = kernels.ForwardStockhamComplex64
	inverseStockhamComplex64  = kernels.InverseStockhamComplex64
	forwardStockhamComplex128 = kernels.ForwardStockhamComplex128
	inverseStockhamComplex128 = kernels.InverseStockhamComplex128
	stockhamForward           = kernels.StockhamForward
	stockhamInverse           = kernels.StockhamInverse

	// Registries
	Registry64  = &kernels.Registry64
	Registry128 = &kernels.Registry128
)

// Wrapper functions for generic functions

func ditForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return kernels.DITForward(dst, src, twiddle, scratch, bitrev)
}

func ditInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return kernels.DITInverse(dst, src, twiddle, scratch, bitrev)
}

func sameSlice[T any](a, b []T) bool {
	return kernels.SameSlice(a, b)
}

func ComputeChirpSequence[T Complex](n int) []T {
	return kernels.ComputeChirpSequence[T](n)
}

func ComputeBluesteinFilter[T Complex](n, m int, chirp []T, twiddles []T, bitrev []int, scratch []T) []T {
	return kernels.ComputeBluesteinFilter[T](n, m, chirp, twiddles, bitrev, scratch)
}

func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T, bitrev []int) {
	kernels.BluesteinConvolution[T](dst, x, filter, twiddles, scratch, bitrev)
}

// Re-export SIMD level constants
const (
	SIMDNone = kernels.SIMDNone
	SIMDSSE2 = kernels.SIMDSSE2
	SIMDAVX2 = kernels.SIMDAVX2
	SIMDNEON = kernels.SIMDNEON
)
