package algofft

import (
	"math/bits"
	"strings"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// Plan is a pre-computed FFT plan for a specific size and precision.
// Plans are reusable and safe for concurrent use during transforms.
//
// The generic type parameter T must be either complex64 or complex128,
// determining the precision of the transform.
type Plan[T Complex] struct {
	// n is the FFT length (number of complex samples).
	n int

	// exec is the strategy-specific engine that runs the transforms. It owns
	// the precomputed tables of exactly one strategy family (see
	// plan_exec*.go) and is shared with clones (immutable after construction).
	exec planExecutor[T]

	// forwardCodelet/inverseCodelet cache the kernel executor's zero-dispatch
	// codelet binding (nil when the plan has none). They exist purely as a
	// Forward/Inverse fast path: one direct call instead of the interface
	// dispatch, which benchstat showed costs ~20ns — dominant for tiny
	// codelet-bound sizes (+70% at n=8). The executor keeps its own copies
	// and is complete without this cache: a codelet that bails here is
	// retried there before the kernel fallback.
	forwardCodelet        fftypes.CodeletFunc[T]
	inverseCodelet        fftypes.CodeletFunc[T]
	codeletTwiddleForward []T
	codeletTwiddleInverse []T

	// algorithm describes which implementation is bound (e.g., "dit64_generic", "stockham").
	algorithm string

	// kernelStrategy is the resolved strategy, reported by introspection and
	// consulted by the strided fast-path gate.
	kernelStrategy fftypes.KernelStrategy

	// twiddle contains the plan's twiddle factors: the standard table
	// W_n^k = exp(-2πik/n) for kernel plans (also used by the strided DIT
	// fast path), the recursive layout for recursive plans, nil for
	// Bluestein/Rader plans (their sub-FFT twiddles live in the executor).
	twiddle []T

	// bitrev contains precomputed radix-2 bit-reversal indices for the
	// strided DIT fast path; nil when that path is unavailable (see
	// planBitReversal).
	bitrev []int

	// twiddleBacking keeps the aligned twiddle memory alive for GC.
	twiddleBacking []byte

	// Fixed per-plan scratch buffers (pooled and cloned plans). When scratch
	// is nil, per-call scratch sets are drawn from scratchPool instead.
	scratch               []T
	stridedScratch        []T
	subScratch            []T // Bluestein/Rader sub-FFT scratch
	scratchBacking        []byte
	stridedScratchBacking []byte
	subScratchBacking     []byte

	// scratchLen/subScratchLen record the scratch sizes the strategy needs,
	// used by scratch-set allocation and Clone.
	scratchLen    int
	subScratchLen int

	// scratchPool manages per-call scratch buffers for thread-safety.
	// Used only when the scratch field is nil.
	scratchPool *scratchCache[T]

	// pool is the buffer pool this Plan was allocated from (nil if not pooled).
	pool *fft.BufferPool
}

// wisdomAdapter adapts the public WisdomStore interface to the internal WisdomRecorder.
type wisdomAdapter struct {
	store WisdomStore
}

func (a wisdomAdapter) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (string, bool) {
	return a.store.LookupWisdom(size, precision, cpuFeatures)
}

func (a wisdomAdapter) Store(entry planner.WisdomEntry) {
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

// planBitReversal precomputes radix-2 bit-reversal indices for power-of-two
// plans so the strided DIT fast path can run without per-call recomputation.
// Non-power-of-two sizes return nil; callers treat a nil table as "fast path
// unavailable".
//
// Recursive plans are excluded: they store recursive-decomposition twiddles in
// p.twiddle (see TwiddleFactorsRecursive), which are incompatible with the
// radix-2 strided DIT fast path in transformStrided. Returning nil keeps that
// fast path disabled for them so strided transforms fall back to the correct
// generic gather/scatter path instead of silently producing a wrong spectrum.
func planBitReversal[T Complex](n int, estimate planner.PlanEstimate[T]) []int {
	if !m.IsPowerOf2(n) || estimate.Strategy == fftypes.KernelRecursive {
		return nil
	}

	return m.ComputeBitReversalIndices(n)
}

// NewPlan creates a new FFT plan for the given size using the generic type T.
// The size n can be any positive integer.
// Power-of-2 sizes are most efficient.
// Highly composite sizes (factors 2, 3, 5) use mixed-radix algorithms, as do
// sizes with factors 7/11 where that measures faster than Bluestein.
// Prime sizes whose n-1 is 5-smooth use Rader's algorithm when it measures
// faster; other primes and remaining sizes use Bluestein's algorithm
// (Chirp-Z transform).
//
// Example:
//
//	plan, err := NewPlan[complex64](1024)
//	plan128, err := NewPlan[complex128](1024)
func NewPlan[T Complex](n int) (*Plan[T], error) {
	return newPlanWithFeatures[T](n, cpu.DetectFeatures(), PlanOptions{})
}

// NewPlanWithOptions creates a new FFT plan with explicit planner options.
func NewPlanWithOptions[T Complex](n int, opts PlanOptions) (*Plan[T], error) {
	return newPlanWithFeatures[T](n, cpu.DetectFeatures(), normalizePlanOptions(opts))
}

// NewPlan32 creates a new single-precision (complex64) FFT plan.
// This is one-line sugar for NewPlan[complex64](n).
func NewPlan32(n int) (*Plan[complex64], error) {
	return NewPlan[complex64](n)
}

// NewPlan64 creates a new double-precision (complex128) FFT plan.
// This is one-line sugar for NewPlan[complex128](n).
func NewPlan64(n int) (*Plan[complex128], error) {
	return NewPlan[complex128](n)
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
	return newPlanFromPoolWithOptions[T](n, fft.DefaultPool, PlanOptions{})
}

// NewPlanPooledWithOptions creates a new FFT plan using pooled buffers and planner options.
//
// It accepts the same lengths and options as NewPlanWithOptions. Sizes or
// forced strategies that require Bluestein or recursive decomposition carry
// extra per-plan tables the shared buffer pool does not manage; those plans
// are built with the regular allocator instead (Close remains valid, it just
// has no buffers to return).
func NewPlanPooledWithOptions[T Complex](n int, opts PlanOptions) (*Plan[T], error) {
	return newPlanFromPoolWithOptions[T](n, fft.DefaultPool, opts)
}

func standardScratchSize(n int, algorithm string) int {
	if strings.Contains(algorithm, "sixstep64x128") {
		return 2 * n
	}

	return n
}

// maxBluesteinLength is the largest transform length a Bluestein plan
// accepts. The padded sub-FFT needs 2n-1 and NextPowerOfTwo(2n-1) to be
// representable in int, which bounds n to 2^61 on 64-bit platforms (2^29 on
// 32-bit) — far beyond any allocatable transform, but without the bound the
// arithmetic in bluesteinPadSize would silently wrap.
const maxBluesteinLength = 1 << (bits.UintSize - 3)

// The padded sub-FFT length for Bluestein plans is chosen by the shared pad
// cost model in plan_padsize.go (bluesteinPadSize / cheapestPaddedLength).

// planStrategyConfig computes the strategy-specific plan configuration: the
// Bluestein padded sub-FFT size (rejecting lengths whose pad size >= 2n-1
// cannot be represented; see maxBluesteinLength), the exact Rader sub-FFT
// size n-1, or the recursive decomposition strategy. Strategies without
// extra configuration return zero values.
//
// Recursive decomposition is limited to power-of-two lengths: the strategy
// tree terminates only in power-of-two codelet/DIT leaves, so any other
// length would build an executor that silently produces a wrong spectrum.
func planStrategyConfig(n int, useBluestein, useRader, useRecursive bool) (int, *transform.DecomposeStrategy, error) {
	switch {
	case useRader:
		// Rader's cyclic convolution runs at exactly n-1 (mixed-radix
		// executable by eligibility), no padding needed.
		return n - 1, nil, nil
	case useBluestein:
		if n > maxBluesteinLength {
			return 0, nil, ErrInvalidLength
		}

		return bluesteinPadSize(n), nil, nil
	case useRecursive:
		if !m.IsPowerOf2(n) {
			return 0, nil, ErrInvalidLength
		}

		codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
		cacheSize := 32768 // L1 cache size estimate

		strategy := transform.PlanDecomposition(n, codeletSizes, cacheSize)
		if strategy == nil {
			return 0, nil, ErrInvalidLength
		}

		return 0, strategy, nil
	default:
		return 0, nil, nil
	}
}

// planScratchSizes returns the main and sub-FFT scratch lengths for a plan
// configuration. The main scratch follows the strategy; the sub-FFT scratch
// exists only for the Bluestein/Rader convolution.
func planScratchSizes(
	n int, strategy fftypes.KernelStrategy, subM int,
	decomp *transform.DecomposeStrategy, algorithm string,
) (int, int) {
	//nolint:exhaustive // only Bluestein/Recursive need non-standard scratch sizes
	switch strategy {
	case fftypes.KernelBluestein:
		// Rader plans set subM = n-1 (exact sub-FFT), so clamp to n: paths
		// outside the convolution (e.g. strided gather) assume the main
		// scratch holds a full length-n frame. Bluestein pads to >= 2n-1,
		// where the clamp is a no-op.
		return max(subM, n), subM
	case fftypes.KernelRecursive:
		return transform.ScratchSizeRecursive(decomp), 0
	default:
		return max(standardScratchSize(n, algorithm), n), 0
	}
}

// algorithmRader is the Algorithm() name reported by prime-length plans that
// run Rader's algorithm (see internal/fft/rader.go and plan_exec_rader.go).
const algorithmRader = "rader"

// convolutionTables groups the precomputed tables shared by the Bluestein and
// Rader arbitrary-length paths. Bluestein fills the chirp sequences; Rader
// fills the generator permutations; both fill the frequency-domain filters,
// sub-FFT twiddles, and (for power-of-two sub-FFT sizes) the bitrev table.
type convolutionTables[T Complex] struct {
	chirp, chirpInv   []T
	filter, filterInv []T
	twiddle           []T
	bitrev            []int
	raderPermIn       []int
	raderPermOut      []int

	// sub is the plan-time binding of the padded power-of-two sub-FFT; nil
	// selects the unbound route. See newBluesteinSubFFT.
	sub *fft.BluesteinSubFFT[T]

	// subTwiddleBacking retains the aligned backing arrays of any prepared
	// codelet twiddle layouts so they stay reachable for the plan's lifetime.
	subTwiddleForwardBacking []byte
	subTwiddleInverseBacking []byte
}

// computeConvolutionTables fills the Rader or Bluestein tables for an
// arbitrary-length plan of size n with sub-FFT size m; other strategies
// return the zero value. Both scratch buffers must have length >= m; probe is
// used only during setup, to verify the candidate sub-FFT binding.
func computeConvolutionTables[T Complex](
	n, m int, useBluestein, useRader bool, features cpu.Features, probe, scratch []T,
) convolutionTables[T] {
	var tables convolutionTables[T]

	switch {
	case useRader:
		tables.raderPermIn, tables.raderPermOut, tables.filter, tables.filterInv,
			tables.twiddle, tables.bitrev = fft.ComputeRaderTables[T](n, scratch)
	case useBluestein:
		tables = computeBluesteinTables[T](n, m, features, probe, scratch)
	}

	return tables
}

// computeBluesteinTables precomputes the chirp sequences, sub-FFT twiddles,
// bit-reversal indices, the bound sub-FFT, and the forward/inverse filters for
// a Bluestein plan of length n with padded sub-FFT size padM.
func computeBluesteinTables[T Complex](
	n, padM int, features cpu.Features, probe, scratch []T,
) convolutionTables[T] {
	var tables convolutionTables[T]

	tables.chirp = kernels.ComputeChirpSequence[T](n)

	tables.chirpInv = make([]T, n)
	for i, v := range tables.chirp {
		tables.chirpInv[i] = m.ConjugateOf(v)
	}

	tables.twiddle = m.ComputeTwiddleFactors[T](padM)

	// bitrev feeds only the unbound power-of-two DIT sub-FFT path; the
	// mixed-radix padded sizes run through the mixed-radix engine, and the
	// bound codelets carry their own layouts.
	if m.IsPowerOf2(padM) {
		tables.bitrev = m.ComputeBitReversalIndices(padM)

		sub, fwdBacking, invBacking := newBluesteinSubFFT[T](padM, tables.twiddle, features, probe, scratch)
		tables.sub = sub
		tables.subTwiddleForwardBacking = fwdBacking
		tables.subTwiddleInverseBacking = invBacking
	}

	// Compute filters using the pre-allocated scratch buffer.
	tables.filter = fft.ComputeBluesteinFilter(n, padM, tables.chirp, tables.twiddle, scratch, tables.sub)
	tables.filterInv = fft.ComputeBluesteinFilter(n, padM, tables.chirpInv, tables.twiddle, scratch, tables.sub)

	return tables
}

// newBluesteinSubFFT binds the padded power-of-two sub-FFT of a Bluestein plan
// to the best kernel the registry and dispatch offer at that size, including a
// codelet with a prepared twiddle layout.
//
// It exists because the unbound route reaches none of them: it enters
// internal/kernels' hardcoded size switch, which never consults the codelet
// registry, so the sub-FFT — ~96% of a Bluestein transform's work — ran in pure
// Go on every build (PLAN.md P3). Binding here rather than at call time is what
// makes a prepared layout possible at all.
//
// The candidate is verified end-to-end before it is installed: a kernel that
// declines the size, or whose prepared layout it rejects, yields a nil binding
// and the caller keeps the unbound route. That check is also what lets the
// transform-time path treat a bail as a contract violation rather than
// attempting a fallback it could not perform safely (dst aliases the input).
func newBluesteinSubFFT[T Complex](
	padM int, twiddle []T, features cpu.Features, probe, scratch []T,
) (*fft.BluesteinSubFFT[T], []byte, []byte) {
	// The sub-FFT is an ordinary power-of-two transform, so it gets the
	// ordinary estimate: registry codelet first, heuristic strategy otherwise.
	// No wisdom store and no forced strategy — the padded size is an internal
	// implementation detail of this plan, not a length the user asked for.
	estimate := planner.EstimatePlan[T](padM, features, nil, fftypes.KernelAuto)

	// Bind only when the registry actually has a codelet for the padded size.
	//
	// Falling back to the strategy-dispatched kernels here would be a
	// regression, not a neutral default: the unbound route's size switch
	// (internal/kernels/dit.go) is hand-tuned per size, while EstimatePlan's
	// heuristic answers Stockham for everything above ditAutoThreshold — so a
	// codelet-less pad of 2048 would trade a tuned radix-4-then-2 DIT for plain
	// Stockham. Measured on -tags purego at n = 1009, that cost ~4%. The point
	// of binding is to reach the registry; where the registry has nothing, the
	// incumbent stays.
	if estimate.ForwardCodelet == nil || estimate.InverseCodelet == nil {
		return nil, nil, nil
	}

	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles(padM, twiddle, estimate)

	sub := &fft.BluesteinSubFFT[T]{
		Forward:        kernels.Kernel[T](estimate.ForwardCodelet),
		Inverse:        kernels.Kernel[T](estimate.InverseCodelet),
		TwiddleForward: twiddleForward,
		TwiddleInverse: twiddleInverse,
	}

	if !fft.VerifyBluesteinSub(sub, padM, probe, scratch) {
		return nil, nil, nil
	}

	return sub, forwardBacking, inverseBacking
}

// selectPlanEstimate chooses the plan estimate according to the planner mode:
// measuring modes micro-benchmark candidate strategies (recording results into
// the Wisdom store when one is configured), while estimate mode uses
// heuristics only.
func selectPlanEstimate[T Complex](n int, features cpu.Features, opts PlanOptions) planner.PlanEstimate[T] {
	switch opts.Planner {
	case PlannerMeasure, PlannerPatient, PlannerExhaustive:
		// Run micro-benchmarks to find the best strategy
		var recorder fft.WisdomRecorder
		if opts.Wisdom != nil {
			recorder = wisdomAdapter{opts.Wisdom}
		}

		return fft.MeasureAndSelect[T](
			n,
			features,
			fft.PlannerMode(opts.Planner),
			recorder,
			opts.Strategy.internal(),
		)
	case PlannerEstimate:
		// PlannerEstimate: use heuristics only (fast path)
		return planner.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy.internal())
	default:
		// Fallback for any unknown planner modes
		return planner.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy.internal())
	}
}

// kernelSelectionStrategy chooses the strategy handed to
// SelectKernelsWithStrategy. Plan estimates pre-resolve KernelAuto (e.g. to
// KernelStockham for large sizes), which would make an auto choice
// indistinguishable from an explicitly forced one at kernel-dispatch time.
// When the user did not force a strategy and the estimate matches the pure
// size heuristic (i.e. it is not a wisdom- or measurement-derived override),
// KernelAuto is passed instead so per-size dispatch keeps the distinction:
// the AVX-512 tier substitutes its faster DIT kernel for auto-resolved
// Stockham sizes while an explicit KernelStockham stays on the Stockham path.
func kernelSelectionStrategy(n int, requested, estimated fftypes.KernelStrategy) fftypes.KernelStrategy {
	if requested == fftypes.KernelAuto && estimated == planner.ResolveKernelStrategy(n) {
		return fftypes.KernelAuto
	}

	return estimated
}

// newKernelExecutor builds the codelet/kernel executor: codelet bindings and
// their twiddle layouts from the plan estimate, the strategy-dispatched
// fallback kernels, and the packed Stockham tables when that route wins at this
// size, precision and instruction-set tier.
func newKernelExecutor[T Complex](
	n int, twiddle []T, kern kernels.Kernels[T], estimate planner.PlanEstimate[T], features cpu.Features,
) *kernelExecutor[T] {
	e := &kernelExecutor[T]{
		forwardCodelet: estimate.ForwardCodelet,
		inverseCodelet: estimate.InverseCodelet,
		twiddle:        twiddle,
		forwardKernel:  kern.Forward,
		inverseKernel:  kern.Inverse,
	}

	e.codeletTwiddleForward, e.codeletTwiddleInverse,
		e.codeletTwiddleForwardBacking, e.codeletTwiddleInverseBacking = prepareCodeletTwiddles(n, twiddle, estimate)

	// The ForwardCodelet check is redundant today and deliberately kept: every
	// registered codelet carries Algorithm: KernelDIT, so an estimate that
	// bound one never reports KernelStockham. It pins that invariant rather
	// than relying on it silently — if a Stockham codelet is ever registered,
	// this keeps the packed table from being allocated behind it.
	if estimate.Strategy == fftypes.KernelStockham && estimate.ForwardCodelet == nil &&
		transform.PackedStockhamEnabled(n, isWideComplex[T](), features) {
		e.packed = transform.ComputePackedTwiddles[T](n, 4, twiddle)
	}

	return e
}

// isWideComplex reports whether T is complex128. It is a single type assertion
// rather than a type switch with concrete bodies in a generic function, which
// Go compiles into every shape instantiation (PLAN.md 2.4).
func isWideComplex[T Complex]() bool {
	_, wide := any(*new(T)).(complex128)

	return wide
}

// newConvolutionExecutor builds the executor for the arbitrary-length
// convolution strategies from the precomputed tables: Rader (exact
// length-(n-1) sub-FFT) when eligible, Bluestein (padded to m) otherwise.
func newConvolutionExecutor[T Complex](n, m int, useRader bool, tables convolutionTables[T]) planExecutor[T] {
	if useRader {
		return &raderExecutor[T]{
			n:         n,
			permIn:    tables.raderPermIn,
			permOut:   tables.raderPermOut,
			filter:    tables.filter,
			filterInv: tables.filterInv,
			twiddle:   tables.twiddle,
			bitrev:    tables.bitrev,
		}
	}

	return &bluesteinExecutor[T]{
		n:         n,
		m:         m,
		chirp:     tables.chirp,
		chirpInv:  tables.chirpInv,
		filter:    tables.filter,
		filterInv: tables.filterInv,
		twiddle:   tables.twiddle,
		bitrev:    tables.bitrev,
		sub:       tables.sub,

		subTwiddleForwardBacking: tables.subTwiddleForwardBacking,
		subTwiddleInverseBacking: tables.subTwiddleInverseBacking,
	}
}

func newPlanWithFeatures[T Complex](n int, features cpu.Features, opts PlanOptions) (*Plan[T], error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}

	estimate := selectPlanEstimate[T](n, features, opts)

	useBluestein := estimate.Strategy == fftypes.KernelBluestein
	useRecursive := estimate.Strategy == fftypes.KernelRecursive
	strategy := estimate.Strategy

	// Prime lengths whose n-1 the mixed-radix engine executes exactly upgrade
	// from Bluestein to Rader's algorithm (exact length-(n-1) convolution
	// instead of one padded to >= 2n-1). An explicitly forced
	// KernelBluestein is honored as-is.
	useRader := useBluestein && opts.Strategy != KernelBluestein && fft.RaderEligible(n)

	bluesteinM, decompStrategy, err := planStrategyConfig(n, useBluestein, useRader, useRecursive)
	if err != nil {
		return nil, err
	}

	// Prewarm shared per-size tables (bit-reversal indices) so the first
	// transform stays allocation-free.
	fft.PrewarmSizeCaches(n)

	// Allocate the initial scratch set for setup (Bluestein/Rader filter
	// computation) and later transforms.
	scratchLen, subScratchLen := planScratchSizes(n, strategy, bluesteinM, decompStrategy, estimate.Algorithm)
	setupScratch := allocateScratchSet[T](n, scratchLen, subScratchLen)

	// setupScratch.scratch doubles as the sub-FFT verification probe during
	// setup: for Bluestein plans planScratchSizes gives it length >= bluesteinM,
	// and no transform has run yet.
	tables := computeConvolutionTables[T](
		n, bluesteinM, useBluestein, useRader, features, setupScratch.scratch, setupScratch.subScratch,
	)

	var (
		twiddle        []T
		twiddleBacking []byte
	)

	switch {
	case useBluestein:
		// Sub-FFT twiddles were computed above alongside the Bluestein/Rader
		// tables and live in the executor; the plan-level table stays nil.
	case useRecursive:
		// Generate twiddles for recursive decomposition.
		twiddle, twiddleBacking = allocTwiddle(transform.TwiddleFactorsRecursive[T](decompStrategy))
	default:
		// Standard allocation.
		twiddle, twiddleBacking = allocTwiddle(m.ComputeTwiddleFactors[T](n))
	}

	algorithm := estimate.Algorithm
	if useRader {
		algorithm = algorithmRader
	}

	var (
		exec planExecutor[T]
		ke   *kernelExecutor[T]
	)

	switch {
	case useBluestein:
		exec = newConvolutionExecutor(n, bluesteinM, useRader, tables)
	case useRecursive:
		exec = &recursiveExecutor[T]{
			strategy: decompStrategy,
			twiddle:  twiddle,
			features: features,
		}
	default:
		// Fallback kernels serve transforms when no codelet is bound (or a
		// codelet bails); only the kernel executor needs them.
		kern := fft.SelectKernelsWithStrategy[T](features, kernelSelectionStrategy(n, opts.Strategy.internal(), strategy))
		ke = newKernelExecutor[T](n, twiddle, kern, estimate, features)
		exec = ke
	}

	// Create the scratch cache and seed it with the setup scratch set.
	scratchPool := new(scratchCache[T])
	scratchPool.pool.New = func() any {
		return allocateScratchSet[T](n, scratchLen, subScratchLen)
	}
	scratchPool.put(setupScratch)

	p := &Plan[T]{
		n:              n,
		exec:           exec,
		algorithm:      algorithm,
		kernelStrategy: strategy,
		twiddle:        twiddle,
		bitrev:         planBitReversal(n, estimate),
		twiddleBacking: twiddleBacking,
		scratchLen:     scratchLen,
		subScratchLen:  subScratchLen,
		scratchPool:    scratchPool,
	}

	if ke != nil {
		p.forwardCodelet = ke.forwardCodelet
		p.inverseCodelet = ke.inverseCodelet
		p.codeletTwiddleForward = ke.codeletTwiddleForward
		p.codeletTwiddleInverse = ke.codeletTwiddleInverse
	}

	return p, nil
}

// newPlanFromPoolWithOptions creates a new FFT plan using buffers from the
// specified pool. It mirrors newPlanWithFeatures' length and planner contract:
// any length newPlanWithFeatures accepts is accepted here, with Bluestein and
// recursive plans delegated to it because their extra tables are not pooled.
func newPlanFromPoolWithOptions[T Complex](n int, pool *fft.BufferPool, opts PlanOptions) (*Plan[T], error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()
	estimate := selectPlanEstimate[T](n, features, opts)

	strategy := estimate.Strategy
	if strategy == fftypes.KernelBluestein || strategy == fftypes.KernelRecursive {
		return newPlanWithFeatures[T](n, features, opts)
	}

	kern := fft.SelectKernelsWithStrategy[T](features, kernelSelectionStrategy(n, opts.Strategy.internal(), strategy))

	// Prewarm shared per-size tables (bit-reversal indices) so the first
	// transform stays allocation-free.
	fft.PrewarmSizeCaches(n)

	scratchLen := max(standardScratchSize(n, estimate.Algorithm), n)
	twiddle, scratch, stridedScratch,
		twiddleBacking, scratchBacking, stridedBacking := getBuffersFromPool[T](n, scratchLen, pool)

	var bitrev []int
	if m.IsPowerOf2(n) {
		bitrev = pool.GetIntSlice(n)
		computed := m.ComputeBitReversalIndices(n)
		copy(bitrev, computed)
	}

	ke := newKernelExecutor[T](n, twiddle, kern, estimate, features)

	return &Plan[T]{
		n:                     n,
		exec:                  ke,
		forwardCodelet:        ke.forwardCodelet,
		inverseCodelet:        ke.inverseCodelet,
		codeletTwiddleForward: ke.codeletTwiddleForward,
		codeletTwiddleInverse: ke.codeletTwiddleInverse,
		algorithm:             estimate.Algorithm,
		kernelStrategy:        strategy,
		twiddle:               twiddle,
		bitrev:                bitrev,
		twiddleBacking:        twiddleBacking,
		scratch:               scratch,
		stridedScratch:        stridedScratch,
		scratchBacking:        scratchBacking,
		stridedScratchBacking: stridedBacking,
		scratchLen:            scratchLen,
		pool:                  pool,
	}, nil
}
