package algoforge

import "github.com/MeKo-Christian/algoforge/internal/fft"

// Plan is a pre-computed FFT plan for a specific size and precision.
// Plans are reusable and safe for concurrent use during transforms.
//
// The generic type parameter T must be either complex64 or complex128,
// determining the precision of the transform.
type Plan[T Complex] struct {
	// n is the FFT length (number of complex samples).
	n int

	// twiddle contains precomputed twiddle factors (roots of unity).
	// For a size-n FFT: W_n^k = exp(-2πik/n) for k = 0..n-1
	twiddle []T

	// scratch is a pre-allocated buffer for intermediate computations.
	// This enables zero-allocation transforms after Plan creation.
	scratch []T

	// bitrev contains precomputed bit-reversal permutation indices.
	// bitrev[i] contains the bit-reversed index for position i.
	bitrev []int

	// packedTwiddle* store prepacked twiddle tables for SIMD-friendly radices.
	packedTwiddle4  *fft.PackedTwiddles[T]
	packedTwiddle8  *fft.PackedTwiddles[T]
	packedTwiddle16 *fft.PackedTwiddles[T]

	forwardKernel  fft.Kernel[T]
	inverseKernel  fft.Kernel[T]
	kernelStrategy fft.KernelStrategy

	// backing buffers keep aligned slices alive for GC.
	twiddleBacking []byte
	scratchBacking []byte
}

// KernelStrategy controls which FFT kernel a plan should use.
type KernelStrategy = fft.KernelStrategy

const (
	KernelAuto     = fft.KernelAuto
	KernelDIT      = fft.KernelDIT
	KernelStockham = fft.KernelStockham
)

// SetKernelStrategy overrides the global kernel selection strategy.
// Use KernelAuto to restore automatic selection.
func SetKernelStrategy(strategy KernelStrategy) {
	fft.SetKernelStrategy(strategy)
}

// GetKernelStrategy returns the current global kernel selection strategy.
func GetKernelStrategy() KernelStrategy {
	return fft.GetKernelStrategy()
}

// RecordBenchmarkDecision stores a per-size kernel choice for auto selection.
func RecordBenchmarkDecision(n int, strategy KernelStrategy) {
	fft.RecordBenchmarkDecision(n, strategy)
}

// Len returns the FFT length (number of complex samples) for this Plan.
func (p *Plan[T]) Len() int {
	return p.n
}

// KernelStrategy reports the strategy chosen when the plan was created.
func (p *Plan[T]) KernelStrategy() KernelStrategy {
	return KernelStrategy(p.kernelStrategy)
}

// Forward computes the forward (time-to-frequency) FFT.
//
// The transform is computed as:
//
//	X[k] = Σ x[n] * exp(-2πink/N) for k = 0..N-1
//
// dst and src must have length equal to Plan.Len().
// dst and src may point to the same slice for in-place operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match Plan dimensions.
func (p *Plan[T]) Forward(dst, src []T) error {
	err := p.validateSlices(dst, src)
	if err != nil {
		return err
	}

	if p.forwardKernel != nil && p.forwardKernel(dst, src, p.twiddle, p.scratch, p.bitrev) {
		return nil
	}

	return ErrNotImplemented
}

// Inverse computes the inverse (frequency-to-time) FFT.
//
// The transform is computed as:
//
//	x[n] = (1/N) * Σ X[k] * exp(2πink/N) for n = 0..N-1
//
// dst and src must have length equal to Plan.Len().
// dst and src may point to the same slice for in-place operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match Plan dimensions.
func (p *Plan[T]) Inverse(dst, src []T) error {
	err := p.validateSlices(dst, src)
	if err != nil {
		return err
	}

	if p.inverseKernel != nil && p.inverseKernel(dst, src, p.twiddle, p.scratch, p.bitrev) {
		return nil
	}

	return ErrNotImplemented
}

// InPlace computes the forward FFT in-place, modifying the input slice directly.
//
// This is equivalent to Forward(data, data) but may be slightly more efficient.
//
// Returns ErrNilSlice if data is nil.
// Returns ErrLengthMismatch if slice length doesn't match Plan dimensions.
func (p *Plan[T]) InPlace(data []T) error {
	return p.Forward(data, data)
}

// InverseInPlace computes the inverse FFT in-place, modifying the input slice directly.
//
// This is equivalent to Inverse(data, data) but may be slightly more efficient.
//
// Returns ErrNilSlice if data is nil.
// Returns ErrLengthMismatch if slice length doesn't match Plan dimensions.
func (p *Plan[T]) InverseInPlace(data []T) error {
	return p.Inverse(data, data)
}

// validateSlices checks that dst and src are valid for this Plan.
func (p *Plan[T]) validateSlices(dst, src []T) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(dst) != p.n || len(src) != p.n {
		return ErrLengthMismatch
	}

	return nil
}

// NewPlan creates a new FFT plan for the given size using the generic type T.
// The size n must be a positive power of 2.
//
// Example:
//
//	plan, err := NewPlan[complex64](1024)
//	plan128, err := NewPlan[complex128](1024)
func NewPlan[T Complex](n int) (*Plan[T], error) {
	if !fft.IsPowerOfTwo(n) || n < 1 {
		return nil, ErrInvalidLength
	}

	features := fft.DetectFeatures()
	strategy := fft.ResolveKernelStrategy(n)
	kernels := fft.SelectKernelsWithStrategy[T](features, strategy)

	var zero T
	var twiddle []T
	var twiddleBacking []byte
	var scratch []T
	var scratchBacking []byte

	switch any(zero).(type) {
	case complex64:
		twiddleAligned, twiddleRaw := fft.AllocAlignedComplex64(n)
		tmp := fft.ComputeTwiddleFactors[complex64](n)
		copy(twiddleAligned, tmp)
		twiddle = any(twiddleAligned).([]T)
		twiddleBacking = twiddleRaw

		scratchAligned, scratchRaw := fft.AllocAlignedComplex64(n)
		scratch = any(scratchAligned).([]T)
		scratchBacking = scratchRaw
	case complex128:
		twiddleAligned, twiddleRaw := fft.AllocAlignedComplex128(n)
		tmp := fft.ComputeTwiddleFactors[complex128](n)
		copy(twiddleAligned, tmp)
		twiddle = any(twiddleAligned).([]T)
		twiddleBacking = twiddleRaw

		scratchAligned, scratchRaw := fft.AllocAlignedComplex128(n)
		scratch = any(scratchAligned).([]T)
		scratchBacking = scratchRaw
	default:
		twiddle = fft.ComputeTwiddleFactors[T](n)
		scratch = make([]T, n)
	}

	p := &Plan[T]{
		n:              n,
		twiddle:        twiddle,
		scratch:        scratch,
		bitrev:         fft.ComputeBitReversalIndices(n),
		forwardKernel:  kernels.Forward,
		inverseKernel:  kernels.Inverse,
		kernelStrategy: strategy,
		twiddleBacking: twiddleBacking,
		scratchBacking: scratchBacking,
	}

	p.packedTwiddle4 = fft.ComputePackedTwiddles[T](n, 4, p.twiddle)
	p.packedTwiddle8 = fft.ComputePackedTwiddles[T](n, 8, p.twiddle)
	p.packedTwiddle16 = fft.ComputePackedTwiddles[T](n, 16, p.twiddle)

	return p, nil
}

// NewPlan32 creates a new single-precision (complex64) FFT plan.
// This is equivalent to NewPlan[complex64](n).
func NewPlan32(n int) (*Plan[complex64], error) {
	return NewPlan[complex64](n)
}

// NewPlan64 creates a new double-precision (complex128) FFT plan.
// This is equivalent to NewPlan[complex128](n).
func NewPlan64(n int) (*Plan[complex128], error) {
	return NewPlan[complex128](n)
}
