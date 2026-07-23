package algofft

import (
	stdmath "math"
	"math/bits"
	"strings"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
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

	// codeletTwiddle* hold codelet-specific twiddle layouts (may alias twiddle).
	codeletTwiddleForward []T
	codeletTwiddleInverse []T

	// scratch is an optional pre-allocated buffer for intermediate computations.
	// If nil, scratch buffers are retrieved from scratchPool per call.
	scratch []T

	// stridedScratch is an optional pre-allocated buffer for strided transforms.
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
	bluesteinM              int   // Padded size M >= 2N-1 (power of two or 5-smooth; see bluesteinPadSize)
	bluesteinChirp          []T   // Size N
	bluesteinChirpInv       []T   // Size N
	bluesteinFilter         []T   // Size M
	bluesteinFilterInv      []T   // Size M
	bluesteinTwiddle        []T   // Size M
	bluesteinBitrev         []int // Size M
	bluesteinScratch        []T   // Size M (extra scratch for Bluestein)
	bluesteinScratchBacking []byte

	// Rader specific fields. A non-nil raderPermIn marks a prime-length plan
	// that runs Rader's algorithm instead of Bluestein; it reuses the
	// bluestein* tables above with M = N-1 (exact, unpadded sub-FFT).
	raderPermIn  []int // Input gather permutation: g^(-q) mod N
	raderPermOut []int // Output scatter permutation: g^m mod N

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
	twiddleBacking               []byte
	codeletTwiddleForwardBacking []byte
	codeletTwiddleInverseBacking []byte
	scratchBacking               []byte
	stridedScratchBacking        []byte

	// pool is the buffer pool this Plan was allocated from (nil if not pooled).
	pool *fft.BufferPool

	// scratchPool manages per-call scratch buffers for thread-safety.
	// Used only when scratch field is nil.
	scratchPool *scratchCache[T]
}

// KernelStrategy controls which FFT kernel a plan should use.
type KernelStrategy = fft.KernelStrategy

const (
	KernelAuto       = fft.KernelAuto       // Let the planner choose by size
	KernelDIT        = fft.KernelDIT        // Decimation-in-time
	KernelStockham   = fft.KernelStockham   // Stockham autosort
	KernelSixStep    = fft.KernelSixStep    // Six-step (cache-friendly, large sizes)
	KernelEightStep  = fft.KernelEightStep  // Eight-step (cache-friendly, large sizes)
	KernelBluestein  = fft.KernelBluestein  // Bluestein (arbitrary lengths)
	KernelRecursive  = fft.KernelRecursive  // Recursive decomposition with codelet leaves
	KernelSplitRadix = fft.KernelSplitRadix // Split-radix (2/4) DIT (power-of-two lengths)
)

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
// Precision names used by the String() methods of all plan types.
const (
	precisionNameComplex64  = "complex64"
	precisionNameComplex128 = "complex128"
)

// Strategy names used by String() and the introspection tests.
const (
	strategyNameDIT       = "DIT"
	strategyNameStockham  = "Stockham"
	strategyNameSixStep   = "SixStep"
	strategyNameEightStep = "EightStep"
)

// The format is: "Plan[type](size, strategy)" where type is "complex64" or "complex128".
func (p *Plan[T]) String() string {
	var zero T

	typeName := precisionNameComplex64

	if _, ok := any(zero).(complex128); ok {
		typeName = precisionNameComplex128
	}

	strategyName := "auto"

	switch p.kernelStrategy {
	case fft.KernelDIT:
		strategyName = strategyNameDIT
	case fft.KernelStockham:
		strategyName = strategyNameStockham
	case fft.KernelSixStep:
		strategyName = strategyNameSixStep
	case fft.KernelEightStep:
		strategyName = strategyNameEightStep
	case fft.KernelBluestein:
		strategyName = "Bluestein"
	case fft.KernelSplitRadix:
		strategyName = "SplitRadix"
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
func planBitReversal[T Complex](n int, estimate fft.PlanEstimate[T]) []int {
	if !m.IsPowerOf2(n) || estimate.Strategy == fft.KernelRecursive {
		return nil
	}

	return fft.ComputeBitReversalIndices(n)
}

func (p *Plan[T]) getScratch() ([]T, []T, []T, *scratchSet[T]) {
	if p.scratch != nil {
		return p.scratch, p.stridedScratch, p.bluesteinScratch, nil
	}

	s := p.scratchPool.get()

	return s.scratch, s.stridedScratch, s.bluesteinScratch, s
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

	scratch, _, bsScratch, set := p.getScratch()
	if set != nil {
		defer p.scratchPool.put(set)
	}

	if p.kernelStrategy == fft.KernelBluestein {
		if p.raderPermIn != nil {
			return p.raderForward(dst, src, scratch, bsScratch)
		}

		return p.bluesteinForward(dst, src, scratch, bsScratch)
	}

	if p.kernelStrategy == fft.KernelRecursive {
		return p.recursiveForward(dst, src, scratch)
	}

	// Zero-dispatch codelet path (highest priority). A codelet reports false
	// when it bailed without doing any work; fall through to the generic
	// dispatch below then instead of returning unset output.
	if p.forwardCodelet != nil && p.forwardCodelet(dst, src, p.codeletTwiddleForward, scratch) {
		return nil
	}

	// Pure-Go packed radix-4 Stockham route. StockhamPackedAvailable is false
	// on SIMD builds, where the codelet path above supersedes it (see the
	// toggle in internal/transform/stockham_packed_toggle_*.go).
	if p.kernelStrategy == fft.KernelStockham && fft.StockhamPackedAvailable() {
		if fft.ForwardStockhamPacked(dst, src, p.twiddle, scratch, p.packedTwiddle4) {
			return nil
		}
	}

	// Fallback kernel dispatch
	if p.forwardKernel != nil && p.forwardKernel(dst, src, p.twiddle, scratch) {
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

	scratch, _, bsScratch, set := p.getScratch()
	if set != nil {
		defer p.scratchPool.put(set)
	}

	if p.kernelStrategy == fft.KernelBluestein {
		if p.raderPermIn != nil {
			return p.raderInverse(dst, src, scratch, bsScratch)
		}

		return p.bluesteinInverse(dst, src, scratch, bsScratch)
	}

	if p.kernelStrategy == fft.KernelRecursive {
		return p.recursiveInverse(dst, src, scratch)
	}

	// Zero-dispatch codelet path (highest priority). A codelet reports false
	// when it bailed without doing any work; fall through to the generic
	// dispatch below then instead of returning unset output.
	if p.inverseCodelet != nil && p.inverseCodelet(dst, src, p.codeletTwiddleInverse, scratch) {
		return nil
	}

	// Pure-Go packed radix-4 Stockham route. StockhamPackedAvailable is false
	// on SIMD builds, where the codelet path above supersedes it (see the
	// toggle in internal/transform/stockham_packed_toggle_*.go).
	if p.kernelStrategy == fft.KernelStockham && fft.StockhamPackedAvailable() {
		if fft.InverseStockhamPacked(dst, src, p.twiddle, scratch, p.packedTwiddle4Inv) {
			return nil
		}
	}

	// Fallback kernel dispatch
	if p.inverseKernel != nil && p.inverseKernel(dst, src, p.twiddle, scratch) {
		return nil
	}

	return ErrNotImplemented
}

// ForwardInPlace computes the forward FFT in-place, modifying the input slice directly.
//
// This is equivalent to Forward(data, data) but may be slightly more efficient.
//
// Returns ErrNilSlice if data is nil.
// Returns ErrLengthMismatch if slice length doesn't match Plan dimensions.
func (p *Plan[T]) ForwardInPlace(data []T) error {
	return p.Forward(data, data)
}

// InPlace computes the forward FFT in-place.
//
// Deprecated: Use ForwardInPlace, which matches the naming of the
// multi-dimensional plans' ForwardInPlace/InverseInPlace pair.
func (p *Plan[T]) InPlace(data []T) error {
	return p.ForwardInPlace(data)
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

// ForwardUnsafe performs the forward FFT without any validation.
// This is a zero-overhead path for latency-critical workloads.
//
// REQUIREMENTS (caller must guarantee):
//   - dst and src are non-nil slices
//   - len(dst) >= p.Len() and len(src) >= p.Len()
//   - Plan has pre-allocated scratch (p.scratch != nil)
//   - Plan has a bound codelet or kernel
//
// Violating these requirements causes undefined behavior or panic.
// Use Forward() for the safe, validated path.
func (p *Plan[T]) ForwardUnsafe(dst, src []T) {
	if p.forwardCodelet != nil && p.forwardCodelet(dst, src, p.codeletTwiddleForward, p.scratch) {
		return
	}

	p.forwardKernel(dst, src, p.twiddle, p.scratch)
}

// InverseUnsafe performs the inverse FFT without any validation.
// This is a zero-overhead path for latency-critical workloads.
//
// REQUIREMENTS (caller must guarantee):
//   - dst and src are non-nil slices
//   - len(dst) >= p.Len() and len(src) >= p.Len()
//   - Plan has pre-allocated scratch (p.scratch != nil)
//   - Plan has a bound codelet or kernel
//
// Violating these requirements causes undefined behavior or panic.
// Use Inverse() for the safe, validated path.
func (p *Plan[T]) InverseUnsafe(dst, src []T) {
	if p.inverseCodelet != nil && p.inverseCodelet(dst, src, p.codeletTwiddleInverse, p.scratch) {
		return
	}

	p.inverseKernel(dst, src, p.twiddle, p.scratch)
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
// Highly composite sizes (factors 2, 3, 5) use mixed-radix algorithms, as do
// sizes with factors 7/11 where that measures faster than Bluestein.
// Prime sizes whose n-1 is 5-smooth use Rader's algorithm when it measures
// faster; other primes and remaining sizes use Bluestein's algorithm
// (Chirp-Z transform).
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

// bluesteinSubFFTPenalty weights the estimated per-point cost of the
// mixed-radix sub-FFT relative to the size-dispatched power-of-two DIT
// sub-FFT. Measured with BenchmarkBluesteinPadCandidates (internal/fft) on
// an AVX2/AVX-512 Xeon: ~2.2x on the purego build and ~4.5x on the default
// (SIMD) build. Because a 5-smooth pad can undercut the next power of two by
// at most ~2x in m·log2(m) work (m is confined to [2n-1, 2·(2n-1))), any
// penalty above ~2.1 means the power of two wins at every size — which is
// what the measurements show for the current kernels, so this constant
// intentionally disables 5-smooth pads. The chooser and the mixed-radix
// sub-FFT path are kept wired up: if the mixed-radix engine gets faster
// (e.g. SIMD radix-3/5 butterflies), re-run the benchmark on both builds and
// lower this constant to re-enable them.
const bluesteinSubFFTPenalty = 2.2

// bluesteinPadSize returns the padded sub-FFT length for a Bluestein plan of
// logical length n. The cyclic convolution needs any length m >= 2n-1 the
// engine can execute exactly; the next 5-smooth size (2^a·3^b·5^c, handled by
// the mixed-radix engine) is frequently much smaller than the next power of
// two (e.g. n=1009: 2025 vs 4096), so both are costed as m·log2(m) with the
// mixed-radix penalty applied and the cheaper one wins. The choice is a pure
// function of n.
func bluesteinPadSize(n int) int {
	minM := 2*n - 1

	pow2 := m.NextPowerOfTwo(minM)

	smooth := m.NextHighlyComposite(minM)
	if smooth >= pow2 || smooth < 2 {
		return pow2
	}

	costPow2 := float64(pow2) * stdmath.Log2(float64(pow2))
	costSmooth := bluesteinSubFFTPenalty * float64(smooth) * stdmath.Log2(float64(smooth))

	if costSmooth < costPow2 {
		return smooth
	}

	return pow2
}

// planStrategyConfig computes the strategy-specific plan configuration: the
// Bluestein padded sub-FFT size (rejecting lengths whose pad size >= 2n-1
// cannot be represented; see maxBluesteinLength), the exact Rader sub-FFT
// size n-1, or the recursive decomposition strategy. Strategies without
// extra configuration return zero values.
func planStrategyConfig(n int, useBluestein, useRader, useRecursive bool) (int, *fft.DecomposeStrategy, error) {
	switch {
	case useRader:
		// Rader's cyclic convolution runs at exactly n-1 (5-smooth by
		// eligibility), no padding needed.
		return n - 1, nil, nil
	case useBluestein:
		if n > maxBluesteinLength {
			return 0, nil, ErrInvalidLength
		}

		return bluesteinPadSize(n), nil, nil
	case useRecursive:
		codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
		cacheSize := 32768 // L1 cache size estimate

		return 0, fft.PlanDecomposition(n, codeletSizes, cacheSize), nil
	default:
		return 0, nil, nil
	}
}

// algorithmRader is the Algorithm() name reported by prime-length plans that
// run Rader's algorithm (see internal/fft/rader.go and plan_rader.go).
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
}

// computeConvolutionTables fills the Rader or Bluestein tables for an
// arbitrary-length plan of size n with sub-FFT size m; other strategies
// return the zero value. scratch must have length >= m.
func computeConvolutionTables[T Complex](n, m int, useBluestein, useRader bool, scratch []T) convolutionTables[T] {
	var tables convolutionTables[T]

	switch {
	case useRader:
		tables.raderPermIn, tables.raderPermOut, tables.filter, tables.filterInv,
			tables.twiddle, tables.bitrev = fft.ComputeRaderTables[T](n, scratch)
	case useBluestein:
		tables.chirp, tables.chirpInv, tables.filter, tables.filterInv,
			tables.twiddle, tables.bitrev = computeBluesteinTables[T](n, m, scratch)
	}

	return tables
}

// computeBluesteinTables precomputes the chirp sequences, sub-FFT twiddles,
// bit-reversal indices, and forward/inverse filters for a Bluestein plan of
// length n with padded sub-FFT size m.
//
//nolint:nonamedreturns // six related tables; names document the tuple
func computeBluesteinTables[T Complex](n, m int, scratch []T) (
	chirp, chirpInv, filter, filterInv, twiddle []T, bitrev []int,
) {
	chirp = fft.ComputeChirpSequence[T](n)

	chirpInv = make([]T, n)
	for i, v := range chirp {
		chirpInv[i] = fft.ConjugateOf(v)
	}

	twiddle = fft.ComputeTwiddleFactors[T](m)

	// bitrev feeds only the power-of-two DIT sub-FFT path; 5-smooth padded
	// sizes run through the mixed-radix engine, which does not use it. (The
	// parameter m shadows the math package alias, so test the bit pattern
	// directly.)
	if m&(m-1) == 0 {
		bitrev = fft.ComputeBitReversalIndices(m)
	}

	// Compute filters using the pre-allocated scratch buffer.
	filter = fft.ComputeBluesteinFilter(n, m, chirp, twiddle, scratch)
	filterInv = fft.ComputeBluesteinFilter(n, m, chirpInv, twiddle, scratch)

	return chirp, chirpInv, filter, filterInv, twiddle, bitrev
}

// selectPlanEstimate chooses the plan estimate according to the planner mode:
// measuring modes micro-benchmark candidate strategies (recording results into
// the Wisdom store when one is configured), while estimate mode uses
// heuristics only.
func selectPlanEstimate[T Complex](n int, features cpu.Features, opts PlanOptions) fft.PlanEstimate[T] {
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
			opts.Strategy,
		)
	case PlannerEstimate:
		// PlannerEstimate: use heuristics only (fast path)
		return fft.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy)
	default:
		// Fallback for any unknown planner modes
		return fft.EstimatePlan[T](n, features, opts.Wisdom, opts.Strategy)
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
func kernelSelectionStrategy(n int, requested, estimated KernelStrategy) KernelStrategy {
	if requested == KernelAuto && estimated == fft.ResolveKernelStrategy(n) {
		return KernelAuto
	}

	return estimated
}

func newPlanWithFeatures[T Complex](n int, features cpu.Features, opts PlanOptions) (*Plan[T], error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}

	estimate := selectPlanEstimate[T](n, features, opts)

	useBluestein := estimate.Strategy == fft.KernelBluestein
	useRecursive := estimate.Strategy == fft.KernelRecursive
	strategy := estimate.Strategy

	// Prime lengths whose n-1 is 5-smooth upgrade from Bluestein to Rader's
	// algorithm (exact length-(n-1) convolution instead of one padded to
	// >= 2n-1). An explicitly forced KernelBluestein is honored as-is.
	useRader := useBluestein && opts.Strategy != fft.KernelBluestein && fft.RaderEligible(n)

	bluesteinM, decompStrategy, err := planStrategyConfig(n, useBluestein, useRader, useRecursive)
	if err != nil {
		return nil, err
	}

	// Get fallback kernels (used when no codelet is available)
	kernels := fft.SelectKernelsWithStrategy[T](features, kernelSelectionStrategy(n, opts.Strategy, strategy))

	// Prewarm shared per-size tables (bit-reversal indices) so the first
	// transform stays allocation-free.
	fft.PrewarmSizeCaches(n)

	var (
		twiddle        []T
		twiddleBacking []byte
	)

	// Allocate initial scratch set for setup (Bluestein/Rader filter computation)
	scratchSize := standardScratchSize(n, estimate.Algorithm)
	setupScratch := allocateScratchSet[T](n, strategy, bluesteinM, decompStrategy, scratchSize)

	tables := computeConvolutionTables[T](n, bluesteinM, useBluestein, useRader, setupScratch.bluesteinScratch)

	switch {
	case useBluestein:
		// Twiddles were computed above alongside the Bluestein/Rader tables.
	case useRecursive:
		// Generate twiddles for recursive decomposition.
		twiddle, twiddleBacking = allocTwiddle(fft.TwiddleFactorsRecursive[T](decompStrategy))
	default:
		// Standard allocation.
		twiddle, twiddleBacking = allocTwiddle(fft.ComputeTwiddleFactors[T](n))
	}

	algorithm := estimate.Algorithm
	if useRader {
		algorithm = algorithmRader
	}

	// Create the scratch cache and seed it with the setup scratch set.
	scratchPool := new(scratchCache[T])
	scratchPool.pool.New = func() any {
		return allocateScratchSet[T](n, strategy, bluesteinM, decompStrategy, scratchSize)
	}
	scratchPool.put(setupScratch)

	p := &Plan[T]{
		n:                     n,
		twiddle:               twiddle,
		codeletTwiddleForward: twiddle,
		codeletTwiddleInverse: twiddle,
		scratch:               nil, // Use pool
		stridedScratch:        nil, // Use pool
		bitrev:                planBitReversal(n, estimate),
		forwardCodelet:        estimate.ForwardCodelet,
		inverseCodelet:        estimate.InverseCodelet,
		algorithm:             algorithm,
		forwardKernel:         kernels.Forward,
		inverseKernel:         kernels.Inverse,
		kernelStrategy:        strategy,
		decompStrategy:        decompStrategy,
		twiddleBacking:        twiddleBacking,
		scratchPool:           scratchPool,
		bluesteinM:            bluesteinM,
		bluesteinChirp:        tables.chirp,
		bluesteinChirpInv:     tables.chirpInv,
		bluesteinFilter:       tables.filter,
		bluesteinFilterInv:    tables.filterInv,
		bluesteinTwiddle:      tables.twiddle,
		bluesteinBitrev:       tables.bitrev,
		bluesteinScratch:      nil, // Use pool
		raderPermIn:           tables.raderPermIn,
		raderPermOut:          tables.raderPermOut,
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

	p.codeletTwiddleForward, p.codeletTwiddleInverse,
		p.codeletTwiddleForwardBacking, p.codeletTwiddleInverseBacking = prepareCodeletTwiddles(n, p.twiddle, estimate)

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
	return newPlanFromPoolWithOptions[T](n, fft.DefaultPool, PlanOptions{})
}

// NewPlanPooledWithOptions creates a new FFT plan using pooled buffers and planner options.
//
// It accepts the same lengths and options as NewPlanTWithOptions. Sizes or
// forced strategies that require Bluestein or recursive decomposition carry
// extra per-plan tables the shared buffer pool does not manage; those plans
// are built with the regular allocator instead (Close remains valid, it just
// has no buffers to return).
func NewPlanPooledWithOptions[T Complex](n int, opts PlanOptions) (*Plan[T], error) {
	return newPlanFromPoolWithOptions[T](n, fft.DefaultPool, opts)
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
	if strategy == fft.KernelBluestein || strategy == fft.KernelRecursive {
		return newPlanWithFeatures[T](n, features, opts)
	}

	kernels := fft.SelectKernelsWithStrategy[T](features, kernelSelectionStrategy(n, opts.Strategy, strategy))

	// Prewarm shared per-size tables (bit-reversal indices) so the first
	// transform stays allocation-free.
	fft.PrewarmSizeCaches(n)

	scratchSize := standardScratchSize(n, estimate.Algorithm)
	twiddle, scratch, stridedScratch,
		twiddleBacking, scratchBacking, stridedBacking := getBuffersFromPool[T](n, scratchSize, pool)

	var bitrev []int
	if m.IsPowerOf2(n) {
		bitrev = pool.GetIntSlice(n)
		computed := fft.ComputeBitReversalIndices(n)
		copy(bitrev, computed)
	}

	p := &Plan[T]{
		n:                     n,
		twiddle:               twiddle,
		codeletTwiddleForward: twiddle,
		codeletTwiddleInverse: twiddle,
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
		scratchPool:           nil, // No internal pool for pooled plans
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

	p.codeletTwiddleForward, p.codeletTwiddleInverse,
		p.codeletTwiddleForwardBacking, p.codeletTwiddleInverseBacking = prepareCodeletTwiddles(n, p.twiddle, estimate)

	return p, nil
}
