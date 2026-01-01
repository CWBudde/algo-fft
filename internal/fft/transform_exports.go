package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/transform"
)

// Re-export transform types for backward compatibility.
type DecomposeStrategy = transform.DecomposeStrategy

// Re-export transform functions (wrappers for generic functions)
// Note: PlanDecomposition is non-generic so can be assigned directly.
var PlanDecomposition = transform.PlanDecomposition

func TwiddleFactorsRecursive[T Complex](strategy *DecomposeStrategy) []T {
	return transform.TwiddleFactorsRecursive[T](strategy)
}

var ScratchSizeRecursive = transform.ScratchSizeRecursive

func RecursiveForward[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) {
	transform.RecursiveForward(dst, src, strategy, twiddle, scratch, registry, features)
}

func RecursiveInverse[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) {
	transform.RecursiveInverse(dst, src, strategy, twiddle, scratch, registry, features)
}

// Re-export PackedTwiddles functions (already aliased in kernels.go).
func ComputePackedTwiddles[T Complex](n, radix int, twiddle []T) *PackedTwiddles[T] {
	return transform.ComputePackedTwiddles[T](n, radix, twiddle)
}

func ConjugatePackedTwiddles[T Complex](packed *PackedTwiddles[T]) *PackedTwiddles[T] {
	return transform.ConjugatePackedTwiddles[T](packed)
}

func ForwardStockhamPacked[T Complex](dst, src, twiddle, scratch []T, packed *PackedTwiddles[T]) bool {
	return transform.ForwardStockhamPacked[T](dst, src, twiddle, scratch, packed)
}

func InverseStockhamPacked[T Complex](dst, src, twiddle, scratch []T, packed *PackedTwiddles[T]) bool {
	return transform.InverseStockhamPacked[T](dst, src, twiddle, scratch, packed)
}
