package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	m "github.com/cwbudde/algo-fft/internal/math"
	mem "github.com/cwbudde/algo-fft/internal/memory"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// scratchSet groups the per-call scratch buffers a Plan hands out, together with
// the []byte backings that keep their aligned memory alive for the GC.
type scratchSet[T any] struct {
	scratch                 []T
	scratchBacking          []byte
	stridedScratch          []T
	stridedScratchBacking   []byte
	bluesteinScratch        []T
	bluesteinScratchBacking []byte
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
	if estimate.TwiddleSize == nil || estimate.PrepareTwiddle == nil {
		return base, base, nil, nil
	}

	twiddleLen := estimate.TwiddleSize(n)
	if twiddleLen <= 0 {
		return base, base, nil, nil
	}

	forward, forwardBacking := mem.AllocAligned[T](twiddleLen)
	inverse, inverseBacking := mem.AllocAligned[T](twiddleLen)

	estimate.PrepareTwiddle(n, false, forward)
	estimate.PrepareTwiddle(n, true, inverse)

	return forward, inverse, forwardBacking, inverseBacking
}

// allocateScratchSet allocates the scratch buffers required by strategy. The
// scratch size follows the strategy (Bluestein → M, Recursive → recursive scratch,
// otherwise the standard size clamped to at least n); the extra Bluestein scratch
// is allocated only for the Bluestein strategy.
func allocateScratchSet[T Complex](
	n int, strategy fftypes.KernelStrategy, bluesteinM int,
	decompStrategy *transform.DecomposeStrategy, standardScratchSize int,
) *scratchSet[T] {
	var scratchSize int

	//nolint:exhaustive // only Bluestein/Recursive need non-standard scratch sizes
	switch strategy {
	case fftypes.KernelBluestein:
		// Rader plans set bluesteinM = n-1 (exact sub-FFT), so clamp to n:
		// paths outside the convolution (e.g. strided gather) assume the
		// standard scratch holds a full length-n frame. Bluestein pads to
		// >= 2n-1, where the clamp is a no-op.
		scratchSize = max(bluesteinM, n)
	case fftypes.KernelRecursive:
		scratchSize = transform.ScratchSizeRecursive(decompStrategy)
	default:
		scratchSize = max(standardScratchSize, n)
	}

	scratch, scratchBacking := mem.AllocAligned[T](scratchSize)
	stridedScratch, stridedBacking := mem.AllocAligned[T](n)

	var (
		bluesteinScratch        []T
		bluesteinScratchBacking []byte
	)

	if strategy == fftypes.KernelBluestein {
		bluesteinScratch, bluesteinScratchBacking = mem.AllocAligned[T](bluesteinM)
	}

	return &scratchSet[T]{
		scratch:                 scratch,
		scratchBacking:          scratchBacking,
		stridedScratch:          stridedScratch,
		stridedScratchBacking:   stridedBacking,
		bluesteinScratch:        bluesteinScratch,
		bluesteinScratchBacking: bluesteinScratchBacking,
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
