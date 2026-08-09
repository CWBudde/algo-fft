package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
	mem "github.com/cwbudde/algo-fft/internal/memory"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// scratchSet groups the per-call scratch buffers a Plan hands out, together with
// the []byte backings that keep their aligned memory alive for the GC.
type scratchSet[T any] struct {
	scratch               []T
	scratchBacking        []byte
	stridedScratch        []T
	stridedScratchBacking []byte
	subScratch            []T // Bluestein/Rader sub-FFT scratch
	subScratchBacking     []byte
}

// scratchCache hands out per-call scratch sets for the 1D Plan; see
// residentCache for the caching strategy.
type scratchCache[T any] = residentCache[scratchSet[T]]

// allocTwiddle copies src into a freshly allocated SIMD-aligned buffer and returns
// it with its backing. Used to move computed twiddle tables onto aligned memory.
func allocTwiddle[T Complex](src []T) ([]T, []byte) {
	dst, backing := mem.AllocAligned[T](len(src))
	copy(dst, src)

	return dst, backing
}

// prepareCodeletTwiddles allocates and fills the forward/inverse codelet twiddle
// layouts when the plan estimate requests them. When no codelet-specific layout is
// needed it returns base for both directions with nil backings.
func prepareCodeletTwiddles[T Complex](
	n int,
	base []T,
	estimate planner.PlanEstimate[T],
) ([]T, []T, []byte, []byte) {
	forward, forwardBacking := prepareCodeletTwiddle(
		n, false, base, estimate.ForwardTwiddleSize, estimate.ForwardPrepareTwiddle,
	)
	inverse, inverseBacking := prepareCodeletTwiddle(
		n, true, base, estimate.InverseTwiddleSize, estimate.InversePrepareTwiddle,
	)

	return forward, inverse, forwardBacking, inverseBacking
}

func prepareCodeletTwiddle[T Complex](
	n int,
	inverse bool,
	base []T,
	twiddleSize registry.TwiddleSizeFunc,
	prepare registry.PrepareTwiddleFunc[T],
) ([]T, []byte) {
	if twiddleSize == nil || prepare == nil {
		return base, nil
	}

	twiddleLen := twiddleSize(n)
	if twiddleLen <= 0 {
		return base, nil
	}

	twiddle, backing := mem.AllocAligned[T](twiddleLen)
	prepare(n, inverse, twiddle)

	return twiddle, backing
}

// allocateScratchSet allocates the per-call scratch buffers for a plan of
// length n: the main scratch (scratchLen), the strided gather/scatter buffer
// (always n), and the Bluestein/Rader sub-FFT scratch (subScratchLen, skipped
// when zero). The lengths come from planScratchSizes.
func allocateScratchSet[T Complex](n, scratchLen, subScratchLen int) *scratchSet[T] {
	scratch, scratchBacking := mem.AllocAligned[T](scratchLen)
	stridedScratch, stridedBacking := mem.AllocAligned[T](n)

	var (
		subScratch        []T
		subScratchBacking []byte
	)

	if subScratchLen > 0 {
		subScratch, subScratchBacking = mem.AllocAligned[T](subScratchLen)
	}

	return &scratchSet[T]{
		scratch:               scratch,
		scratchBacking:        scratchBacking,
		stridedScratch:        stridedScratch,
		stridedScratchBacking: stridedBacking,
		subScratch:            subScratch,
		subScratchBacking:     subScratchBacking,
	}
}

// getBuffersFromPool draws the twiddle, scratch, and strided-scratch buffers for a
// pooled plan from pool, filling the twiddle table with the computed factors.
//
//nolint:nonamedreturns // six return values need names to be readable
func getBuffersFromPool[T Complex](n, scratchSize int, pool *fft.BufferPool) (
	twiddle, scratch, stridedScratch []T, twiddleBacking, scratchBacking, stridedBacking []byte,
) {
	if scratchSize < n {
		scratchSize = n
	}

	twiddle, twiddleBacking = fft.PoolGet[T](pool, n)
	copy(twiddle, m.ComputeTwiddleFactors[T](n))

	scratch, scratchBacking = fft.PoolGet[T](pool, scratchSize)
	stridedScratch, stridedBacking = fft.PoolGet[T](pool, n)

	return twiddle, scratch, stridedScratch, twiddleBacking, scratchBacking, stridedBacking
}
