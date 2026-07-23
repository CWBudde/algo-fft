package algofft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
	mem "github.com/cwbudde/algo-fft/internal/memory"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// FastPlan provides zero-overhead FFT transforms for latency-critical workloads.
// All validation and dispatch is resolved at creation time.
//
// Unlike Plan[T], FastPlan:
//   - Always has pre-allocated scratch (no sync.Pool)
//   - Always has direct codelet bindings (no fallback dispatch)
//   - Performs no validation on Forward/Inverse calls
//
// Use NewFastPlan to create instances. Returns ErrNotImplemented if no
// codelet is available for the requested size.
type FastPlan[T Complex] struct {
	n                            int
	twiddle                      []T
	codeletTwiddleForward        []T
	codeletTwiddleInverse        []T
	scratch                      []T
	twiddleBacking               []byte
	codeletTwiddleForwardBacking []byte
	codeletTwiddleInverseBacking []byte
	scratchBacking               []byte

	forwardFunc fft.CodeletFunc[T]
	inverseFunc fft.CodeletFunc[T]

	algorithm string
	strategy  fft.KernelStrategy
}

// NewFastPlan creates an optimized FFT plan with pre-resolved dispatch.
// Returns ErrNotImplemented if no codelet is registered for this size.
//
// Example:
//
//	plan, err := algofft.NewFastPlan[complex64](256)
//	if err != nil {
//	    // Fall back to regular Plan for this size
//	    regularPlan, _ := algofft.NewPlan[complex64](256)
//	}
func NewFastPlan[T Complex](n int) (*FastPlan[T], error) {
	if n < 1 || !m.IsPowerOf2(n) {
		return nil, ErrInvalidLength
	}

	features := cpu.DetectFeatures()
	estimate := planner.EstimatePlan[T](n, features, nil, fft.KernelAuto)

	// Require codelets - no fallback dispatch
	if estimate.ForwardCodelet == nil || estimate.InverseCodelet == nil {
		return nil, ErrNotImplemented
	}

	// Allocate aligned twiddle factors
	var (
		twiddle        []T
		scratch        []T
		twiddleBacking []byte
		scratchBacking []byte
		zero           T
	)

	switch any(zero).(type) {
	case complex64:
		tw, twb := mem.AllocAlignedComplex64(n)
		sc, scb := mem.AllocAlignedComplex64(n)
		tmp := m.ComputeTwiddleFactors[complex64](n)
		copy(tw, tmp)
		twiddle = any(tw).([]T)
		scratch = any(sc).([]T)
		twiddleBacking = twb
		scratchBacking = scb
	case complex128:
		tw, twb := mem.AllocAlignedComplex128(n)
		sc, scb := mem.AllocAlignedComplex128(n)
		tmp := m.ComputeTwiddleFactors[complex128](n)
		copy(tw, tmp)
		twiddle = any(tw).([]T)
		scratch = any(sc).([]T)
		twiddleBacking = twb
		scratchBacking = scb
	}

	fp := &FastPlan[T]{
		n:                     n,
		twiddle:               twiddle,
		codeletTwiddleForward: twiddle,
		codeletTwiddleInverse: twiddle,
		scratch:               scratch,
		twiddleBacking:        twiddleBacking,
		scratchBacking:        scratchBacking,
		forwardFunc:           estimate.ForwardCodelet,
		inverseFunc:           estimate.InverseCodelet,
		algorithm:             estimate.Algorithm,
		strategy:              estimate.Strategy,
	}

	fp.codeletTwiddleForward, fp.codeletTwiddleInverse,
		fp.codeletTwiddleForwardBacking, fp.codeletTwiddleInverseBacking = prepareCodeletTwiddles(n, twiddle, estimate)

	return fp, nil
}

// Len returns the FFT size.
func (fp *FastPlan[T]) Len() int {
	return fp.n
}

// String returns a human-readable description of the FastPlan for debugging.
func (fp *FastPlan[T]) String() string {
	var zero T

	typeName := precisionNameComplex64
	if _, ok := any(zero).(complex128); ok {
		typeName = precisionNameComplex128
	}

	return "FastPlan[" + typeName + "](" + itoa(fp.n) + ", " + fp.algorithm + ")"
}

// Clone creates an independent copy of the FastPlan with its own scratch
// buffer. Clones share the immutable twiddle tables and codelet bindings, so
// each goroutine can transform concurrently on its own clone.
func (fp *FastPlan[T]) Clone() *FastPlan[T] {
	var (
		scratch        []T
		scratchBacking []byte
		zero           T
	)

	switch any(zero).(type) {
	case complex64:
		sc, scb := mem.AllocAlignedComplex64(fp.n)
		scratch = any(sc).([]T)
		scratchBacking = scb
	case complex128:
		sc, scb := mem.AllocAlignedComplex128(fp.n)
		scratch = any(sc).([]T)
		scratchBacking = scb
	}

	clone := *fp
	clone.scratch = scratch
	clone.scratchBacking = scratchBacking

	return &clone
}

// Forward performs the forward FFT without validation.
// Caller guarantees: len(dst) >= n, len(src) >= n, slices non-nil.
func (fp *FastPlan[T]) Forward(dst, src []T) {
	if !fp.forwardFunc(dst, src, fp.codeletTwiddleForward, fp.scratch) {
		panic("algofft: FastPlan codelet rejected its input (caller contract violated?)")
	}
}

// Inverse performs the inverse FFT without validation.
// Caller guarantees: len(dst) >= n, len(src) >= n, slices non-nil.
func (fp *FastPlan[T]) Inverse(dst, src []T) {
	if !fp.inverseFunc(dst, src, fp.codeletTwiddleInverse, fp.scratch) {
		panic("algofft: FastPlan codelet rejected its input (caller contract violated?)")
	}
}

// ForwardInPlace performs the forward FFT in-place without validation.
// Caller guarantees: len(data) >= n, slice non-nil.
func (fp *FastPlan[T]) ForwardInPlace(data []T) {
	if !fp.forwardFunc(data, data, fp.codeletTwiddleForward, fp.scratch) {
		panic("algofft: FastPlan codelet rejected its input (caller contract violated?)")
	}
}

// InverseInPlace performs the inverse FFT in-place without validation.
// Caller guarantees: len(data) >= n, slice non-nil.
func (fp *FastPlan[T]) InverseInPlace(data []T) {
	if !fp.inverseFunc(data, data, fp.codeletTwiddleInverse, fp.scratch) {
		panic("algofft: FastPlan codelet rejected its input (caller contract violated?)")
	}
}
