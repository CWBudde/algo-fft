package algofft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/fft"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
	mem "github.com/MeKo-Christian/algo-fft/internal/memory"
)

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

	// stridedScratch is a pre-allocated buffer for strided transforms.
	stridedScratch []T

	// bitrev contains precomputed bit-reversal permutation indices.
	// bitrev[i] contains the bit-reversed index for position i.
	bitrev []int

	// packedTwiddle* store prepacked twiddle tables for SIMD-friendly radices.
	packedTwiddle4    *fft.PackedTwiddles[T]
	packedTwiddle4Inv *fft.PackedTwiddles[T]
	packedTwiddle8    *fft.PackedTwiddles[T]
	packedTwiddle16   *fft.PackedTwiddles[T]

	// Bluestein specific fields (used only if kernelStrategy == KernelBluestein)
	bluesteinM              int   // Padded size M >= 2N-1
	bluesteinChirp          []T   // Size N
	bluesteinChirpInv       []T   // Size N
	bluesteinFilter         []T   // Size M
	bluesteinFilterInv      []T   // Size M
	bluesteinTwiddle        []T   // Size M
	bluesteinBitrev         []int // Size M
	bluesteinScratch        []T   // Size M (extra scratch for Bluestein)
	bluesteinScratchBacking []byte

	// Zero-dispatch codelet bindings (nil = use fallback kernel)
	forwardCodelet fft.CodeletFunc[T]
	inverseCodelet fft.CodeletFunc[T]

	// algorithm describes which implementation is bound (e.g., "dit64_generic", "stockham")
	algorithm string

	forwardKernel  fft.Kernel[T]
	inverseKernel  fft.Kernel[T]
	kernelStrategy fft.KernelStrategy
	meta           PlanMeta

	// Recursive decomposition strategy (nil if using existing kernel path)
	decompStrategy *fft.DecomposeStrategy

	// backing buffers keep aligned slices alive for GC.
	twiddleBacking        []byte
	scratchBacking        []byte
	stridedScratchBacking []byte

	// pool is the buffer pool this Plan was allocated from (nil if not pooled).
	pool *fft.BufferPool
}

// KernelStrategy controls which FFT kernel a plan should use.
type KernelStrategy = fft.KernelStrategy

const (
	KernelAuto      = fft.KernelAuto
	KernelDIT       = fft.KernelDIT
	KernelStockham  = fft.KernelStockham
	KernelSixStep   = fft.KernelSixStep
	KernelEightStep = fft.KernelEightStep
	KernelBluestein = fft.KernelBluestein
	KernelRecursive = fft.KernelRecursive // Recursive decomposition with codelet leaves
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

// wisdomAdapter adapts the public WisdomStore interface to the internal WisdomRecorder.
type wisdomAdapter struct {
	store WisdomStore
}

func (a wisdomAdapter) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (string, bool) {
	return a.store.LookupWisdom(size, precision, cpuFeatures)
}

func (a wisdomAdapter) Store(entry fft.WisdomEntry) {
	a.store.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        entry.Key.Size,
			Precision:   entry.Key.Precision,
			CPUFeatures: entry.Key.CPUFeatures,
		},
		Algorithm: entry.Algorithm,
		Timestamp: entry.Timestamp,
	})
}

// Len returns the FFT length (number of complex samples) for this Plan.
func (p *Plan[T]) Len() int {
	return p.n
}

// KernelStrategy reports the strategy chosen when the plan was created.
func (p *Plan[T]) KernelStrategy() KernelStrategy {
	return p.kernelStrategy
}

// Algorithm returns the name of the bound kernel or codelet (e.g., "dit8_generic").
// Returns empty string if no specific algorithm is bound.
func (p *Plan[T]) Algorithm() string {
	return p.algorithm
}

// String returns a human-readable description of the Plan for debugging.
// The format is: "Plan[type](size, strategy)" where type is "complex64" or "complex128".
func (p *Plan[T]) String() string {
	var zero T

	typeName := "complex64"

	if _, ok := any(zero).(complex128); ok {
		typeName = "complex128"
	}

	strategyName := "auto"

	switch p.kernelStrategy {
	case fft.KernelDIT:
		strategyName = "DIT"
	case fft.KernelStockham:
		strategyName = "Stockham"
	case fft.KernelSixStep:
		strategyName = "SixStep"
	case fft.KernelEightStep:
		strategyName = "EightStep"
	case fft.KernelBluestein:
		strategyName = "Bluestein"
	}

	pooled := ""
	if p.pool != nil {
		pooled = ", pooled"
	}

	return "Plan[" + typeName + "](" + itoa(p.n) + ", " + strategyName + pooled + ")"
}

// itoa converts an int to a string without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}

	negative := n < 0
	if negative {
		n = -n
	}

	var buf [20]byte

	i := len(buf)

	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}

	if negative {
		i--
		buf[i] = '-'
	}

	return string(buf[i:])
}

func planBitReversal[T Complex](n int, estimate fft.PlanEstimate[T]) []int {
	if !m.IsPowerOf2(n) {
		return nil
	}

	// Use codelet-specific bit-reversal if available
	if estimate.BitrevFunc != nil {
		return estimate.BitrevFunc(n)
	}

	// Fallback to radix-2 binary reversal for kernels without codelets
	return fft.ComputeBitReversalIndices(n)
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

	if p.kernelStrategy == fft.KernelBluestein {
		return p.bluesteinForward(dst, src)
	}

	if p.kernelStrategy == fft.KernelRecursive {
		return p.recursiveForward(dst, src)
	}

	// Zero-dispatch codelet path (highest priority)
	if p.forwardCodelet != nil {
		p.forwardCodelet(dst, src, p.twiddle, p.scratch, p.bitrev)
		return nil
	}

	if p.kernelStrategy == fft.KernelStockham && fft.StockhamPackedAvailable() {
		if fft.ForwardStockhamPacked(dst, src, p.twiddle, p.scratch, p.packedTwiddle4) {
			return nil
		}
	}

	// Fallback kernel dispatch
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

	if p.kernelStrategy == fft.KernelBluestein {
		return p.bluesteinInverse(dst, src)
	}

	if p.kernelStrategy == fft.KernelRecursive {
		return p.recursiveInverse(dst, src)
	}

	// Zero-dispatch codelet path (highest priority)
	if p.inverseCodelet != nil {
		p.inverseCodelet(dst, src, p.twiddle, p.scratch, p.bitrev)
		return nil
	}

	if p.kernelStrategy == fft.KernelStockham && fft.StockhamPackedAvailable() {
		if fft.InverseStockhamPacked(dst, src, p.twiddle, p.scratch, p.packedTwiddle4Inv) {
			return nil
		}
	}

	// Fallback kernel dispatch
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

// Transform computes either forward or inverse FFT based on the inverse flag.
// This is a convenience wrapper over Forward/Inverse.
func (p *Plan[T]) Transform(dst, src []T, inverse bool) error {
	if inverse {
		return p.Inverse(dst, src)
	}

	return p.Forward(dst, src)
}

// ForwardBatch computes count forward FFTs on sequential data.
//
// The data layout is interleaved/sequential:
//   - FFT 0: src[0:n] → dst[0:n]
//   - FFT 1: src[n:2n] → dst[n:2n]
//   - FFT i: src[i*n:(i+1)*n] → dst[i*n:(i+1)*n]
//
// dst and src must have length >= count * Plan.Len().
// dst and src may point to the same slice for in-place batch operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidLength if count < 1.
// Returns ErrLengthMismatch if slice lengths are insufficient.
func (p *Plan[T]) ForwardBatch(dst, src []T, count int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if count < 1 {
		return ErrInvalidLength
	}

	required := count * p.n
	if len(dst) < required || len(src) < required {
		return ErrLengthMismatch
	}

	for i := range count {
		start := i * p.n

		end := start + p.n

		err := p.Forward(dst[start:end], src[start:end])
		if err != nil {
			return err
		}
	}

	return nil
}

// InverseBatch computes count inverse FFTs on sequential data.
//
// The data layout is interleaved/sequential:
//   - FFT 0: src[0:n] → dst[0:n]
//   - FFT 1: src[n:2n] → dst[n:2n]
//   - FFT i: src[i*n:(i+1)*n] → dst[i*n:(i+1)*n]
//
// dst and src must have length >= count * Plan.Len().
// dst and src may point to the same slice for in-place batch operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidLength if count < 1.
// Returns ErrLengthMismatch if slice lengths are insufficient.
func (p *Plan[T]) InverseBatch(dst, src []T, count int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if count < 1 {
		return ErrInvalidLength
	}

	required := count * p.n
	if len(dst) < required || len(src) < required {
		return ErrLengthMismatch
	}

	for i := range count {
		start := i * p.n

		end := start + p.n

		err := p.Inverse(dst[start:end], src[start:end])
		if err != nil {
			return err
		}
	}

	return nil
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

// NewPlanT creates a new FFT plan for the given size using the generic type T.
// The size n can be any positive integer.
// Power-of-2 sizes are most efficient.
// Highly composite sizes (factors 2, 3, 5) use mixed-radix algorithms.
// Prime or other sizes use Bluestein's algorithm (Chirp-Z transform).
//
// Example:
//
//	plan, err := NewPlanT[complex64](1024)
//	plan128, err := NewPlanT[complex128](1024)
func NewPlanT[T Complex](n int) (*Plan[T], error) {
	return newPlanWithFeatures[T](n, cpu.DetectFeatures(), PlanOptions{})
}

// NewPlanWithOptions creates a new FFT plan with explicit planner options.
func NewPlanWithOptions[T Complex](n int, opts PlanOptions) (*Plan[T], error) {
	return newPlanWithFeatures[T](n, cpu.DetectFeatures(), normalizePlanOptions(opts))
}

func newPlanWithFeatures[T Complex](n int, features cpu.Features, opts PlanOptions) (*Plan[T], error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}

	// Choose planning strategy based on mode
	var estimate fft.PlanEstimate[T]

	switch opts.Planner {
	case PlannerMeasure, PlannerPatient, PlannerExhaustive:
		// Run micro-benchmarks to find the best strategy
		var recorder fft.WisdomRecorder
		if opts.Wisdom != nil {
			recorder = wisdomAdapter{opts.Wisdom}
		}

		estimate = fft.MeasureAndSelect[T](
			n,
			features,
			fft.PlannerMode(opts.Planner),
			recorder,
			opts.Strategy,
		)
	case PlannerEstimate:
		// PlannerEstimate: use heuristics only (fast path)
		estimate = fft.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy)
	default:
		// Fallback for any unknown planner modes
		estimate = fft.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy)
	}

	useBluestein := estimate.Strategy == fft.KernelBluestein
	useRecursive := estimate.Strategy == fft.KernelRecursive
	strategy := estimate.Strategy

	// Get fallback kernels (used when no codelet is available)
	kernels := fft.SelectKernelsWithStrategy[T](features, strategy)

	var (
		zero           T
		twiddle        []T
		twiddleBacking []byte
		scratch        []T
		scratchBacking []byte
		stridedScratch []T
		stridedBacking []byte

		// Bluestein specific
		bluesteinM              int
		bluesteinChirp          []T
		bluesteinChirpInv       []T
		bluesteinFilter         []T
		bluesteinFilterInv      []T
		bluesteinTwiddle        []T
		bluesteinBitrev         []int
		bluesteinScratch        []T
		bluesteinScratchBacking []byte

		// Recursive decomposition specific
		decompStrategy *fft.DecomposeStrategy
	)

	if useBluestein {
		bluesteinM = m.NextPowerOfTwo(2*n - 1)
		scratchSize := bluesteinM

		// Alloc scratch (size M)
		switch any(zero).(type) {
		case complex64:
			scratchAligned, scratchRaw := mem.AllocAlignedComplex64(scratchSize)
			scratch = any(scratchAligned).([]T) //nolint:forcetypeassert
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := mem.AllocAlignedComplex64(n)
			stridedScratch = any(stridedAligned).([]T) //nolint:forcetypeassert
			stridedBacking = stridedRaw

			bsAligned, bsRaw := mem.AllocAlignedComplex64(scratchSize)
			bluesteinScratch = any(bsAligned).([]T) //nolint:forcetypeassert
			bluesteinScratchBacking = bsRaw
		case complex128:
			scratchAligned, scratchRaw := mem.AllocAlignedComplex128(scratchSize)
			scratch = any(scratchAligned).([]T) //nolint:forcetypeassert
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := mem.AllocAlignedComplex128(n)
			stridedScratch = any(stridedAligned).([]T) //nolint:forcetypeassert
			stridedBacking = stridedRaw

			bsAligned, bsRaw := mem.AllocAlignedComplex128(scratchSize)
			bluesteinScratch = any(bsAligned).([]T) //nolint:forcetypeassert
			bluesteinScratchBacking = bsRaw
		default:
			scratch = make([]T, scratchSize)
			stridedScratch = make([]T, n)
			bluesteinScratch = make([]T, scratchSize)
		}

		// Compute Bluestein tables
		bluesteinChirp = fft.ComputeChirpSequence[T](n)

		bluesteinChirpInv = make([]T, n)
		for i, v := range bluesteinChirp {
			bluesteinChirpInv[i] = fft.ConjugateOf(v)
		}

		bluesteinTwiddle = fft.ComputeTwiddleFactors[T](bluesteinM)
		bluesteinBitrev = fft.ComputeBitReversalIndices(bluesteinM)

		// Compute filters using the pre-allocated scratch buffer
		bluesteinFilter = fft.ComputeBluesteinFilter(n, bluesteinM, bluesteinChirp, bluesteinTwiddle, bluesteinBitrev, bluesteinScratch)
		bluesteinFilterInv = fft.ComputeBluesteinFilter(n, bluesteinM, bluesteinChirpInv, bluesteinTwiddle, bluesteinBitrev, bluesteinScratch)
	} else if useRecursive {
		// Recursive decomposition: plan decomposition and generate specialized twiddles
		codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
		cacheSize := 32768 // L1 cache size estimate
		decompStrategy = fft.PlanDecomposition(n, codeletSizes, cacheSize)

		// Generate twiddles for recursive decomposition
		var twiddleSize int

		switch any(zero).(type) {
		case complex64:
			tmpTwiddle := fft.TwiddleFactorsRecursive[complex64](decompStrategy)
			twiddleSize = len(tmpTwiddle)
			twiddleAligned, twiddleRaw := mem.AllocAlignedComplex64(twiddleSize)
			copy(twiddleAligned, tmpTwiddle)
			twiddle = any(twiddleAligned).([]T)
			twiddleBacking = twiddleRaw

			scratchSize := fft.ScratchSizeRecursive(decompStrategy)
			scratchAligned, scratchRaw := mem.AllocAlignedComplex64(scratchSize)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := mem.AllocAlignedComplex64(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw
		case complex128:
			tmpTwiddle := fft.TwiddleFactorsRecursive[complex128](decompStrategy)
			twiddleSize = len(tmpTwiddle)
			twiddleAligned, twiddleRaw := mem.AllocAlignedComplex128(twiddleSize)
			copy(twiddleAligned, tmpTwiddle)
			twiddle = any(twiddleAligned).([]T)
			twiddleBacking = twiddleRaw

			scratchSize := fft.ScratchSizeRecursive(decompStrategy)
			scratchAligned, scratchRaw := mem.AllocAlignedComplex128(scratchSize)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := mem.AllocAlignedComplex128(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw
		default:
			twiddle = fft.TwiddleFactorsRecursive[T](decompStrategy)
			scratch = make([]T, fft.ScratchSizeRecursive(decompStrategy))
			stridedScratch = make([]T, n)
		}
	} else {
		// Standard allocation
		switch any(zero).(type) {
		case complex64:
			twiddleAligned, twiddleRaw := mem.AllocAlignedComplex64(n)
			tmp := fft.ComputeTwiddleFactors[complex64](n)
			copy(twiddleAligned, tmp)
			twiddle = any(twiddleAligned).([]T)
			twiddleBacking = twiddleRaw

			scratchAligned, scratchRaw := mem.AllocAlignedComplex64(n)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := mem.AllocAlignedComplex64(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw
		case complex128:
			twiddleAligned, twiddleRaw := mem.AllocAlignedComplex128(n)
			tmp := fft.ComputeTwiddleFactors[complex128](n)
			copy(twiddleAligned, tmp)
			twiddle = any(twiddleAligned).([]T)
			twiddleBacking = twiddleRaw

			scratchAligned, scratchRaw := mem.AllocAlignedComplex128(n)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := mem.AllocAlignedComplex128(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw
		default:
			twiddle = fft.ComputeTwiddleFactors[T](n)
			scratch = make([]T, n)
			stridedScratch = make([]T, n)
		}
	}

	p := &Plan[T]{
		n:                       n,
		twiddle:                 twiddle,
		scratch:                 scratch,
		stridedScratch:          stridedScratch,
		bitrev:                  planBitReversal(n, estimate),
		forwardCodelet:          estimate.ForwardCodelet,
		inverseCodelet:          estimate.InverseCodelet,
		algorithm:               estimate.Algorithm,
		forwardKernel:           kernels.Forward,
		inverseKernel:           kernels.Inverse,
		kernelStrategy:          strategy,
		decompStrategy:          decompStrategy,
		twiddleBacking:          twiddleBacking,
		scratchBacking:          scratchBacking,
		stridedScratchBacking:   stridedBacking,
		bluesteinM:              bluesteinM,
		bluesteinChirp:          bluesteinChirp,
		bluesteinChirpInv:       bluesteinChirpInv,
		bluesteinFilter:         bluesteinFilter,
		bluesteinFilterInv:      bluesteinFilterInv,
		bluesteinTwiddle:        bluesteinTwiddle,
		bluesteinBitrev:         bluesteinBitrev,
		bluesteinScratch:        bluesteinScratch,
		bluesteinScratchBacking: bluesteinScratchBacking,
		meta: PlanMeta{
			Planner:  opts.Planner,
			Strategy: strategy,
			Batch:    opts.Batch,
			Stride:   opts.Stride,
			InPlace:  opts.InPlace,
		},
	}

	if !useBluestein {
		p.packedTwiddle4 = fft.ComputePackedTwiddles[T](n, 4, p.twiddle)
		p.packedTwiddle4Inv = fft.ConjugatePackedTwiddles(p.packedTwiddle4)
		p.packedTwiddle8 = fft.ComputePackedTwiddles[T](n, 8, p.twiddle)
		p.packedTwiddle16 = fft.ComputePackedTwiddles[T](n, 16, p.twiddle)
	}

	return p, nil
}

// NewPlan creates a new single-precision (complex64) FFT plan.
// This is equivalent to NewPlan32(n).
func NewPlan(n int) (*Plan[complex64], error) {
	return NewPlan32(n)
}

// NewPlan32 creates a new single-precision (complex64) FFT plan.
// This is equivalent to NewPlanT[complex64](n).
func NewPlan32(n int) (*Plan[complex64], error) {
	return NewPlanT[complex64](n)
}

// NewPlan64 creates a new double-precision (complex128) FFT plan.
// This is equivalent to NewPlanT[complex128](n).
func NewPlan64(n int) (*Plan[complex128], error) {
	return NewPlanT[complex128](n)
}

// NewPlanPooled creates a new FFT plan using pooled buffer allocations.
// This is more efficient when creating and destroying many Plans of the same size.
//
// The returned Plan should be closed with Close() when no longer needed to return
// buffers to the pool. If Close() is not called, the buffers will eventually be
// garbage collected, but reuse efficiency will be reduced.
//
// Example:
//
//	plan, err := NewPlanPooled[complex64](1024)
//	defer plan.Close()
func NewPlanPooled[T Complex](n int) (*Plan[T], error) {
	return NewPlanFromPool[T](n, fft.DefaultPool)
}

// NewPlanPooledWithOptions creates a new FFT plan using pooled buffers and planner options.
func NewPlanPooledWithOptions[T Complex](n int, opts PlanOptions) (*Plan[T], error) {
	return NewPlanFromPoolWithOptions[T](n, fft.DefaultPool, opts)
}

// NewPlanFromPool creates a new FFT plan using buffers from the specified pool.
// This allows custom pool management for advanced use cases.
func NewPlanFromPool[T Complex](n int, pool *fft.BufferPool) (*Plan[T], error) {
	return NewPlanFromPoolWithOptions[T](n, pool, PlanOptions{})
}

// NewPlanFromPoolWithOptions creates a new FFT plan using buffers from the specified pool and planner options.
func NewPlanFromPoolWithOptions[T Complex](n int, pool *fft.BufferPool, opts PlanOptions) (*Plan[T], error) {
	if n < 1 || (!m.IsPowerOf2(n) && !m.IsHighlyComposite(n)) {
		return nil, ErrInvalidLength
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()
	estimate := fft.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy)

	strategy := estimate.Strategy
	if strategy == fft.KernelBluestein {
		return nil, ErrNotImplemented
	}

	kernels := fft.SelectKernelsWithStrategy[T](features, strategy)

	twiddle, scratch, stridedScratch, twiddleBacking, scratchBacking, stridedBacking := getBuffersFromPool[T](n, pool)

	var bitrev []int
	if m.IsPowerOf2(n) {
		bitrev = pool.GetIntSlice(n)
		computed := fft.ComputeBitReversalIndices(n)
		copy(bitrev, computed)
	}

	p := &Plan[T]{
		n:                     n,
		twiddle:               twiddle,
		scratch:               scratch,
		stridedScratch:        stridedScratch,
		bitrev:                bitrev,
		forwardCodelet:        estimate.ForwardCodelet,
		inverseCodelet:        estimate.InverseCodelet,
		algorithm:             estimate.Algorithm,
		forwardKernel:         kernels.Forward,
		inverseKernel:         kernels.Inverse,
		kernelStrategy:        strategy,
		twiddleBacking:        twiddleBacking,
		scratchBacking:        scratchBacking,
		stridedScratchBacking: stridedBacking,
		pool:                  pool,
		meta: PlanMeta{
			Planner:  opts.Planner,
			Strategy: strategy,
			Batch:    opts.Batch,
			Stride:   opts.Stride,
			InPlace:  opts.InPlace,
		},
	}

	p.packedTwiddle4 = fft.ComputePackedTwiddles[T](n, 4, p.twiddle)
	p.packedTwiddle4Inv = fft.ConjugatePackedTwiddles(p.packedTwiddle4)
	p.packedTwiddle8 = fft.ComputePackedTwiddles[T](n, 8, p.twiddle)
	p.packedTwiddle16 = fft.ComputePackedTwiddles[T](n, 16, p.twiddle)

	return p, nil
}

func getBuffersFromPool[T Complex](n int, pool *fft.BufferPool) (twiddle, scratch, stridedScratch []T, twiddleBacking, scratchBacking, stridedBacking []byte) {
	var zero T
	switch any(zero).(type) {
	case complex64:
		twiddleAligned, twiddleRaw := pool.GetComplex64(n)
		tmp := fft.ComputeTwiddleFactors[complex64](n)
		copy(twiddleAligned, tmp)
		twiddle = any(twiddleAligned).([]T)
		twiddleBacking = twiddleRaw

		scratchAligned, scratchRaw := pool.GetComplex64(n)
		scratch = any(scratchAligned).([]T)
		scratchBacking = scratchRaw

		stridedAligned, stridedRaw := pool.GetComplex64(n)
		stridedScratch = any(stridedAligned).([]T)
		stridedBacking = stridedRaw
	case complex128:
		twiddleAligned, twiddleRaw := pool.GetComplex128(n)
		tmp := fft.ComputeTwiddleFactors[complex128](n)
		copy(twiddleAligned, tmp)
		twiddle = any(twiddleAligned).([]T)
		twiddleBacking = twiddleRaw

		scratchAligned, scratchRaw := pool.GetComplex128(n)
		scratch = any(scratchAligned).([]T)
		scratchBacking = scratchRaw

		stridedAligned, stridedRaw := pool.GetComplex128(n)
		stridedScratch = any(stridedAligned).([]T)
		stridedBacking = stridedRaw
	default:
		twiddle = fft.ComputeTwiddleFactors[T](n)
		scratch = make([]T, n)
		stridedScratch = make([]T, n)
	}

	return twiddle, scratch, stridedScratch, twiddleBacking, scratchBacking, stridedBacking
}

// Reset clears the scratch buffer and resets internal state.
// This can be useful to ensure deterministic behavior or to clear sensitive data.
// The Plan remains usable after Reset.
func (p *Plan[T]) Reset() {
	// Clear scratch buffer
	clear(p.scratch)
	clear(p.stridedScratch)
}

// Close releases pooled resources back to the buffer pool.
// After Close, the Plan must not be used.
//
// Close is only necessary for Plans created with NewPlanPooled or NewPlanFromPool.
// For Plans created with NewPlan, Close is a no-op.
//
// It is safe to call Close multiple times; subsequent calls are no-ops.
func (p *Plan[T]) Close() {
	if p.pool == nil {
		return // Not a pooled plan
	}

	var zero T
	switch any(zero).(type) {
	case complex64:
		if p.twiddleBacking != nil {
			p.pool.PutComplex64(p.n, any(p.twiddle).([]complex64), p.twiddleBacking)
		}

		if p.scratchBacking != nil {
			p.pool.PutComplex64(p.n, any(p.scratch).([]complex64), p.scratchBacking)
		}

		if p.stridedScratchBacking != nil {
			p.pool.PutComplex64(p.n, any(p.stridedScratch).([]complex64), p.stridedScratchBacking)
		}
	case complex128:
		if p.twiddleBacking != nil {
			p.pool.PutComplex128(p.n, any(p.twiddle).([]complex128), p.twiddleBacking)
		}

		if p.scratchBacking != nil {
			p.pool.PutComplex128(p.n, any(p.scratch).([]complex128), p.scratchBacking)
		}

		if p.stridedScratchBacking != nil {
			p.pool.PutComplex128(p.n, any(p.stridedScratch).([]complex128), p.stridedScratchBacking)
		}
	}

	if p.bitrev != nil {
		p.pool.PutIntSlice(p.n, p.bitrev)
	}

	// Clear references to prevent reuse after Close
	p.pool = nil
	p.twiddle = nil
	p.scratch = nil
	p.bitrev = nil
	p.twiddleBacking = nil
	p.scratchBacking = nil
}

// Clone creates an independent copy of the Plan with its own scratch buffer.
// This is useful when multiple goroutines need to perform transforms concurrently,
// as each goroutine should use its own Plan to avoid data races on the scratch buffer.
//
// The cloned Plan shares immutable data (twiddle factors, bit-reversal indices)
// with the original for memory efficiency, but has its own scratch buffer.
//
// Cloned Plans are never pooled, even if the original was.
// Calling Close() on a cloned Plan is a no-op.
func (p *Plan[T]) Clone() *Plan[T] {
	var (
		zero                    T
		scratch                 []T
		scratchBacking          []byte
		bluesteinScratch        []T
		bluesteinScratchBacking []byte
	)

	scratchSize := p.n
	if p.kernelStrategy == fft.KernelBluestein {
		scratchSize = p.bluesteinM
	}

	switch any(zero).(type) {
	case complex64:
		scratchAligned, scratchRaw := mem.AllocAlignedComplex64(scratchSize)
		scratch = any(scratchAligned).([]T)
		scratchBacking = scratchRaw

		if p.kernelStrategy == fft.KernelBluestein {
			bsAligned, bsRaw := mem.AllocAlignedComplex64(p.bluesteinM)
			bluesteinScratch = any(bsAligned).([]T)
			bluesteinScratchBacking = bsRaw
		}
	case complex128:
		scratchAligned, scratchRaw := mem.AllocAlignedComplex128(scratchSize)
		scratch = any(scratchAligned).([]T)
		scratchBacking = scratchRaw

		if p.kernelStrategy == fft.KernelBluestein {
			bsAligned, bsRaw := mem.AllocAlignedComplex128(p.bluesteinM)
			bluesteinScratch = any(bsAligned).([]T)
			bluesteinScratchBacking = bsRaw
		}
	default:
		scratch = make([]T, scratchSize)
		if p.kernelStrategy == fft.KernelBluestein {
			bluesteinScratch = make([]T, p.bluesteinM)
		}
	}

	return &Plan[T]{
		n:                 p.n,
		twiddle:           p.twiddle,           // Shared (immutable)
		scratch:           scratch,             // New allocation
		bitrev:            p.bitrev,            // Shared (immutable)
		packedTwiddle4:    p.packedTwiddle4,    // Shared (immutable)
		packedTwiddle4Inv: p.packedTwiddle4Inv, // Shared (immutable)
		packedTwiddle8:    p.packedTwiddle8,    // Shared (immutable)
		packedTwiddle16:   p.packedTwiddle16,   // Shared (immutable)
		forwardCodelet:    p.forwardCodelet,    // Shared (function pointer)
		inverseCodelet:    p.inverseCodelet,    // Shared (function pointer)
		algorithm:         p.algorithm,         // Shared (immutable string)
		forwardKernel:     p.forwardKernel,
		inverseKernel:     p.inverseKernel,
		kernelStrategy:    p.kernelStrategy,
		meta:              p.meta,
		twiddleBacking:    p.twiddleBacking, // Shared reference (keeps original alive)
		scratchBacking:    scratchBacking,   // New allocation
		pool:              nil,              // Clones are never pooled

		// Bluestein fields
		bluesteinM:              p.bluesteinM,
		bluesteinChirp:          p.bluesteinChirp,
		bluesteinChirpInv:       p.bluesteinChirpInv,
		bluesteinFilter:         p.bluesteinFilter,
		bluesteinFilterInv:      p.bluesteinFilterInv,
		bluesteinTwiddle:        p.bluesteinTwiddle,
		bluesteinBitrev:         p.bluesteinBitrev,
		bluesteinScratch:        bluesteinScratch,        // New allocation
		bluesteinScratchBacking: bluesteinScratchBacking, // New allocation
	}
}
