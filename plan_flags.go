package algoforge

import (
	"time"

	"github.com/MeKo-Christian/algoforge/internal/cpu"
	"github.com/MeKo-Christian/algoforge/internal/fft"
)

// PlanFlags controls planning behavior.
type PlanFlags uint32

const (
	// FlagEstimate uses heuristics to select kernels (fast, default).
	FlagEstimate PlanFlags = 0

	// FlagUseWisdom checks the wisdom cache before planning.
	FlagUseWisdom PlanFlags = 1 << 0

	// FlagSaveWisdom stores the planning decision in the wisdom cache.
	FlagSaveWisdom PlanFlags = 1 << 1

	// FlagDefaultPlanning is the default set of flags for NewPlan.
	// It uses estimation and wisdom for optimal performance.
	FlagDefaultPlanning = FlagEstimate | FlagUseWisdom | FlagSaveWisdom
)

// NewPlanWithFlags creates a new FFT plan with explicit control over planning behavior.
//
// Flags control how the kernel selection is performed:
//   - FlagEstimate: Use heuristics (fast, default)
//   - FlagUseWisdom: Check wisdom cache before planning
//   - FlagSaveWisdom: Store decision in wisdom cache
//
// Example:
//
//	plan, err := NewPlanWithFlags[complex64](1024, FlagUseWisdom | FlagSaveWisdom)
func NewPlanWithFlags[T Complex](n int, flags PlanFlags) (*Plan[T], error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}

	features := cpu.DetectFeatures()

	// Check wisdom cache if requested
	var wisdomEntry fft.WisdomEntry
	var foundWisdom bool

	if flags&FlagUseWisdom != 0 {
		key := fft.MakeWisdomKey[T](n, features.HasSSE2, features.HasAVX2, features.HasAVX512, features.HasNEON)
		wisdomEntry, foundWisdom = fft.DefaultWisdom.Lookup(key)
	}

	// Get plan estimate (tries codelet registry first)
	estimate := fft.EstimatePlan[T](n, features)

	// If wisdom found, try to use the cached algorithm
	if foundWisdom && wisdomEntry.Algorithm != "" {
		// Try to find the codelet by signature
		registry := fft.GetRegistry[T]()
		if registry != nil {
			entry := registry.LookupBySignature(n, wisdomEntry.Algorithm)
			if entry != nil {
				estimate.ForwardCodelet = entry.Forward
				estimate.InverseCodelet = entry.Inverse
				estimate.Algorithm = entry.Signature
				estimate.Strategy = entry.Algorithm
			}
		}
	}

	// Create the plan
	plan, err := createPlanWithEstimate[T](n, features, estimate)
	if err != nil {
		return nil, err
	}

	// Save to wisdom cache if requested
	if flags&FlagSaveWisdom != 0 && estimate.Algorithm != "" {
		key := fft.MakeWisdomKey[T](n, features.HasSSE2, features.HasAVX2, features.HasAVX512, features.HasNEON)
		fft.DefaultWisdom.Store(fft.WisdomEntry{
			Key:       key,
			Algorithm: estimate.Algorithm,
			Timestamp: time.Now(),
		})
	}

	return plan, nil
}

// createPlanWithEstimate creates a plan using the given estimate.
// This is an internal helper that encapsulates the common plan creation logic.
func createPlanWithEstimate[T Complex](n int, features cpu.Features, estimate fft.PlanEstimate[T]) (*Plan[T], error) {
	useBluestein := estimate.Strategy == fft.KernelBluestein
	strategy := estimate.Strategy

	// Get fallback kernels
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
	)

	if useBluestein {
		bluesteinM = fft.NextPowerOfTwo(2*n - 1)
		scratchSize := bluesteinM

		switch any(zero).(type) {
		case complex64:
			scratchAligned, scratchRaw := fft.AllocAlignedComplex64(scratchSize)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := fft.AllocAlignedComplex64(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw

			bsAligned, bsRaw := fft.AllocAlignedComplex64(scratchSize)
			bluesteinScratch = any(bsAligned).([]T)
			bluesteinScratchBacking = bsRaw
		case complex128:
			scratchAligned, scratchRaw := fft.AllocAlignedComplex128(scratchSize)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := fft.AllocAlignedComplex128(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw

			bsAligned, bsRaw := fft.AllocAlignedComplex128(scratchSize)
			bluesteinScratch = any(bsAligned).([]T)
			bluesteinScratchBacking = bsRaw
		default:
			scratch = make([]T, scratchSize)
			stridedScratch = make([]T, n)
			bluesteinScratch = make([]T, scratchSize)
		}

		bluesteinChirp = fft.ComputeChirpSequence[T](n)

		bluesteinChirpInv = make([]T, n)
		for i, v := range bluesteinChirp {
			bluesteinChirpInv[i] = fft.ConjugateOf(v)
		}

		bluesteinTwiddle = fft.ComputeTwiddleFactors[T](bluesteinM)
		bluesteinBitrev = fft.ComputeBitReversalIndices(bluesteinM)

		bluesteinFilter = fft.ComputeBluesteinFilter(n, bluesteinM, bluesteinChirp, bluesteinTwiddle, bluesteinBitrev, bluesteinScratch)
		bluesteinFilterInv = fft.ComputeBluesteinFilter(n, bluesteinM, bluesteinChirpInv, bluesteinTwiddle, bluesteinBitrev, bluesteinScratch)
	} else {
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

			stridedAligned, stridedRaw := fft.AllocAlignedComplex64(n)
			stridedScratch = any(stridedAligned).([]T)
			stridedBacking = stridedRaw
		case complex128:
			twiddleAligned, twiddleRaw := fft.AllocAlignedComplex128(n)
			tmp := fft.ComputeTwiddleFactors[complex128](n)
			copy(twiddleAligned, tmp)
			twiddle = any(twiddleAligned).([]T)
			twiddleBacking = twiddleRaw

			scratchAligned, scratchRaw := fft.AllocAlignedComplex128(n)
			scratch = any(scratchAligned).([]T)
			scratchBacking = scratchRaw

			stridedAligned, stridedRaw := fft.AllocAlignedComplex128(n)
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
		bitrev:                  planBitReversal(n),
		forwardCodelet:          estimate.ForwardCodelet,
		inverseCodelet:          estimate.InverseCodelet,
		forwardKernel:           kernels.Forward,
		inverseKernel:           kernels.Inverse,
		kernelStrategy:          strategy,
		algorithm:               estimate.Algorithm,
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
	}

	if !useBluestein {
		p.packedTwiddle4 = fft.ComputePackedTwiddles[T](n, 4, p.twiddle)
		p.packedTwiddle8 = fft.ComputePackedTwiddles[T](n, 8, p.twiddle)
		p.packedTwiddle16 = fft.ComputePackedTwiddles[T](n, 16, p.twiddle)
	}

	return p, nil
}
