package algoforge

import (
	"time"

	"github.com/MeKo-Christian/algoforge/internal/cpu"
	"github.com/MeKo-Christian/algoforge/internal/fft"
)

// PlanFlags controls planning behavior.
type PlanFlags uint32

const (
	// FlagUseWisdom checks the wisdom cache before planning.
	FlagUseWisdom PlanFlags = 1 << 0

	// FlagSaveWisdom stores the planning decision in the wisdom cache.
	FlagSaveWisdom PlanFlags = 1 << 1

	// FlagDefaultPlanning is the recommended set of flags for most use cases.
	// It enables wisdom cache lookup and storage for optimal performance.
	// To disable wisdom, use 0 (zero value) or construct flags manually.
	FlagDefaultPlanning = FlagUseWisdom | FlagSaveWisdom
)

// NewPlanWithFlags creates a new FFT plan with explicit control over planning behavior.
//
// Flags control how the kernel selection is performed:
//   - 0 (zero): Use heuristics only, no wisdom cache (fast planning)
//   - FlagUseWisdom: Check wisdom cache before planning
//   - FlagSaveWisdom: Store decision in wisdom cache
//   - FlagDefaultPlanning: Recommended combination (FlagUseWisdom | FlagSaveWisdom)
//
// Example:
//
//	plan, err := NewPlanWithFlags[complex64](1024, FlagDefaultPlanning)
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
			} else {
				// Wisdom might point to a fallback kernel strategy.
				// Apply it over the default estimate.
				switch wisdomEntry.Algorithm {
				case "dit_fallback":
					estimate.Strategy = fft.KernelDIT
					estimate.Algorithm = "dit_fallback"
				case "stockham":
					estimate.Strategy = fft.KernelStockham
					estimate.Algorithm = "stockham"
				case "sixstep":
					estimate.Strategy = fft.KernelSixStep
					estimate.Algorithm = "sixstep"
				case "eightstep":
					estimate.Strategy = fft.KernelEightStep
					estimate.Algorithm = "eightstep"
				case "bluestein":
					estimate.Strategy = fft.KernelBluestein
					estimate.Algorithm = "bluestein"
				}
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

// planBuffers holds allocated buffers for a plan.
type planBuffers[T Complex] struct {
	twiddle        []T
	twiddleBacking []byte
	scratch        []T
	scratchBacking []byte
	stridedScratch []T
	stridedBacking []byte
}

// bluesteinBuffers holds Bluestein-specific buffers.
type bluesteinBuffers[T Complex] struct {
	m              int
	chirp          []T
	chirpInv       []T
	filter         []T
	filterInv      []T
	twiddle        []T
	bitrev         []int
	scratch        []T
	scratchBacking []byte
}

// allocateStandardBuffers allocates buffers for standard power-of-two FFTs.
func allocateStandardBuffers[T Complex](n int) planBuffers[T] {
	var zero T
	switch any(zero).(type) {
	case complex64:
		twiddleAligned, twiddleRaw := fft.AllocAlignedComplex64(n)
		tmp := fft.ComputeTwiddleFactors[complex64](n)
		copy(twiddleAligned, tmp)

		scratchAligned, scratchRaw := fft.AllocAlignedComplex64(n)
		stridedAligned, stridedRaw := fft.AllocAlignedComplex64(n)

		return planBuffers[T]{
			twiddle:        any(twiddleAligned).([]T),
			twiddleBacking: twiddleRaw,
			scratch:        any(scratchAligned).([]T),
			scratchBacking: scratchRaw,
			stridedScratch: any(stridedAligned).([]T),
			stridedBacking: stridedRaw,
		}
	case complex128:
		twiddleAligned, twiddleRaw := fft.AllocAlignedComplex128(n)
		tmp := fft.ComputeTwiddleFactors[complex128](n)
		copy(twiddleAligned, tmp)

		scratchAligned, scratchRaw := fft.AllocAlignedComplex128(n)
		stridedAligned, stridedRaw := fft.AllocAlignedComplex128(n)

		return planBuffers[T]{
			twiddle:        any(twiddleAligned).([]T),
			twiddleBacking: twiddleRaw,
			scratch:        any(scratchAligned).([]T),
			scratchBacking: scratchRaw,
			stridedScratch: any(stridedAligned).([]T),
			stridedBacking: stridedRaw,
		}
	default:
		return planBuffers[T]{
			twiddle:        fft.ComputeTwiddleFactors[T](n),
			scratch:        make([]T, n),
			stridedScratch: make([]T, n),
		}
	}
}

// allocateBluesteinBuffers allocates buffers for Bluestein's algorithm.
func allocateBluesteinBuffers[T Complex](n int) bluesteinBuffers[T] {
	m := fft.NextPowerOfTwo(2*n - 1)

	var zero T
	var bluesteinScratch []T
	var bluesteinScratchBacking []byte

	switch any(zero).(type) {
	case complex64:
		bsAligned, bsRaw := fft.AllocAlignedComplex64(m)
		bluesteinScratch = any(bsAligned).([]T)
		bluesteinScratchBacking = bsRaw
	case complex128:
		bsAligned, bsRaw := fft.AllocAlignedComplex128(m)
		bluesteinScratch = any(bsAligned).([]T)
		bluesteinScratchBacking = bsRaw
	default:
		bluesteinScratch = make([]T, m)
	}

	chirp := fft.ComputeChirpSequence[T](n)
	chirpInv := make([]T, n)
	for i, v := range chirp {
		chirpInv[i] = fft.ConjugateOf(v)
	}

	twiddle := fft.ComputeTwiddleFactors[T](m)
	bitrev := fft.ComputeBitReversalIndices(m)

	filter := fft.ComputeBluesteinFilter(n, m, chirp, twiddle, bitrev, bluesteinScratch)
	filterInv := fft.ComputeBluesteinFilter(n, m, chirpInv, twiddle, bitrev, bluesteinScratch)

	return bluesteinBuffers[T]{
		m:              m,
		chirp:          chirp,
		chirpInv:       chirpInv,
		filter:         filter,
		filterInv:      filterInv,
		twiddle:        twiddle,
		bitrev:         bitrev,
		scratch:        bluesteinScratch,
		scratchBacking: bluesteinScratchBacking,
	}
}

// createPlanWithEstimate creates a plan using the given estimate.
// This is an internal helper that encapsulates the common plan creation logic.
func createPlanWithEstimate[T Complex](n int, features cpu.Features, estimate fft.PlanEstimate[T]) (*Plan[T], error) {
	useBluestein := estimate.Strategy == fft.KernelBluestein
	strategy := estimate.Strategy

	// Get fallback kernels
	kernels := fft.SelectKernelsWithStrategy[T](features, strategy)

	var buffers planBuffers[T]
	var bluestein bluesteinBuffers[T]

	if useBluestein {
		bluestein = allocateBluesteinBuffers[T](n)
		// For Bluestein, use the scratch buffers from the Bluestein allocation
		buffers.scratch = bluestein.scratch
		buffers.scratchBacking = bluestein.scratchBacking
	} else {
		buffers = allocateStandardBuffers[T](n)
	}

	p := &Plan[T]{
		n:                       n,
		twiddle:                 buffers.twiddle,
		scratch:                 buffers.scratch,
		stridedScratch:          buffers.stridedScratch,
		bitrev:                  planBitReversal(n),
		forwardCodelet:          estimate.ForwardCodelet,
		inverseCodelet:          estimate.InverseCodelet,
		forwardKernel:           kernels.Forward,
		inverseKernel:           kernels.Inverse,
		kernelStrategy:          strategy,
		algorithm:               estimate.Algorithm,
		twiddleBacking:          buffers.twiddleBacking,
		scratchBacking:          buffers.scratchBacking,
		stridedScratchBacking:   buffers.stridedBacking,
		bluesteinM:              bluestein.m,
		bluesteinChirp:          bluestein.chirp,
		bluesteinChirpInv:       bluestein.chirpInv,
		bluesteinFilter:         bluestein.filter,
		bluesteinFilterInv:      bluestein.filterInv,
		bluesteinTwiddle:        bluestein.twiddle,
		bluesteinBitrev:         bluestein.bitrev,
		bluesteinScratch:        bluestein.scratch,
		bluesteinScratchBacking: bluestein.scratchBacking,
	}

	if !useBluestein {
		p.packedTwiddle4 = fft.ComputePackedTwiddles[T](n, 4, p.twiddle)
		p.packedTwiddle8 = fft.ComputePackedTwiddles[T](n, 8, p.twiddle)
		p.packedTwiddle16 = fft.ComputePackedTwiddles[T](n, 16, p.twiddle)
	}

	return p, nil
}
