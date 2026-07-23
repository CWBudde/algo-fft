package algofft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/registry"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// recursiveExecutor runs power-of-two transforms via recursive decomposition
// with codelet leaves. The constructor guarantees a non-nil strategy tree
// (non-power-of-two lengths are rejected with ErrInvalidLength).
//
// The per-precision type switches are value-dispatch bridges into the
// monomorphized transform entry points; they are scheduled for removal with
// the generated complex128 kernel twins (PLAN.md A5).
type recursiveExecutor[T Complex] struct {
	strategy *transform.DecomposeStrategy

	// twiddle holds the recursive-decomposition twiddle layout (see
	// transform.TwiddleFactorsRecursive), shared with the owning Plan.
	twiddle []T

	// features are captured at plan construction so transforms skip the
	// per-call CPU feature lookup.
	features cpu.Features
}

func (e *recursiveExecutor[T]) forward(dst, src, scratch, _ []T) {
	switch any(dst).(type) {
	case []complex64:
		transform.RecursiveForward(
			any(dst).([]complex64), any(src).([]complex64), e.strategy,
			any(e.twiddle).([]complex64), any(scratch).([]complex64),
			registry.Registry64, e.features,
		)
	case []complex128:
		transform.RecursiveForward(
			any(dst).([]complex128), any(src).([]complex128), e.strategy,
			any(e.twiddle).([]complex128), any(scratch).([]complex128),
			registry.Registry128, e.features,
		)
	default:
		panic("algofft: internal error: unsupported element type in recursive executor")
	}
}

func (e *recursiveExecutor[T]) inverse(dst, src, scratch, _ []T) {
	switch any(dst).(type) {
	case []complex64:
		transform.RecursiveInverse(
			any(dst).([]complex64), any(src).([]complex64), e.strategy,
			any(e.twiddle).([]complex64), any(scratch).([]complex64),
			registry.Registry64, e.features,
		)
	case []complex128:
		transform.RecursiveInverse(
			any(dst).([]complex128), any(src).([]complex128), e.strategy,
			any(e.twiddle).([]complex128), any(scratch).([]complex128),
			registry.Registry128, e.features,
		)
	default:
		panic("algofft: internal error: unsupported element type in recursive executor")
	}
}
