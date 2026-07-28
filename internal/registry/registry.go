// Package registry holds the codelet registry: size-indexed lookup of
// hand-tuned FFT codelets. It is a leaf package (importing only fftypes and
// cpu) so that layering stays acyclic: internal/kernels registers codelets
// into it at init time, and internal/planner and internal/transform read from
// it during plan estimation and recursive execution.
package registry

import (
	"maps"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
)

// TwiddleSizeFunc returns the element count needed for codelet twiddles.
// Returns 0 if the codelet uses standard twiddles.
type TwiddleSizeFunc func(n int) int

// PrepareTwiddleFunc fills codelet-specific twiddle data.
// It receives the size, inverse flag, and destination slice.
// The destination slice is owned by the Plan and persists for its lifetime.
type PrepareTwiddleFunc[T fftypes.Complex] func(n int, inverse bool, dst []T)

// CodeletEntry describes a registered codelet for a specific size.
type CodeletEntry[T fftypes.Complex] struct {
	Size       int                    // FFT size this codelet handles
	Forward    fftypes.CodeletFunc[T] // Forward transform (nil if not available)
	Inverse    fftypes.CodeletFunc[T] // Inverse transform (nil if not available)
	Algorithm  fftypes.KernelStrategy // DIT, Stockham, etc.
	SIMDLevel  fftypes.SIMDLevel      // Required CPU features
	Signature  string                 // Human-readable name: "dit8_avx2"
	Priority   int                    // Higher priority = preferred (within a rank level)
	KernelType fftypes.KernelType     // How the kernel handles permutation

	// RankLevel overrides the SIMD level used for *ordering* only; SIMDLevel
	// alone still decides whether a CPU may run the codelet. Zero means "rank
	// at SIMDLevel", which is the case for every codelet that does not need
	// this.
	//
	// The two are separate because a codelet's instruction set is a poor proxy
	// for its speed. Some AVX2-encoded codelets are SSE-width in practice
	// (they never fill a YMM register), so they belong in the SSE2 tier's
	// ranking even though they fault on a pre-AVX2 CPU. Without this, the
	// SIMD-level-dominates-priority rule below makes them unbeatable by a
	// genuinely faster SSE2 sibling no matter how the priorities are set —
	// which silently defeated the hand-tuned priorities on the complex128
	// n = 64 codelets for as long as they carried a "stay below
	// dit64_radix4_sse2" comment.
	//
	// Set it to demote (rank a wide-ISA codelet into a narrower tier) rather
	// than to promote: promoting a narrow codelet also moves it ahead of its
	// own tier's siblings on CPUs that have nothing better.
	RankLevel fftypes.SIMDLevel

	// Codelet twiddle preparation (nil = use standard twiddle layout)
	TwiddleSize    TwiddleSizeFunc       // Returns element count for codelet twiddles
	PrepareTwiddle PrepareTwiddleFunc[T] // Prepares twiddle layout for the codelet
}

// rank returns the SIMD level this entry is ordered by. An unset RankLevel
// (fftypes.SIMDNone, the zero value) means "rank at SIMDLevel"; ranking a
// codelet explicitly into the generic tier is therefore not expressible, which
// is fine because a generic codelet already ranks there.
func (e *CodeletEntry[T]) rank() fftypes.SIMDLevel {
	if e.RankLevel == fftypes.SIMDNone {
		return e.SIMDLevel
	}

	return e.RankLevel
}

// CodeletRegistry provides size-indexed codelet lookup.
// Codelets are organized by size, with multiple implementations per size
// (e.g., generic, AVX2, NEON variants).
//
// The size map is copy-on-write: Register publishes a fresh map under mu, and
// readers take an atomic snapshot without locking. Registration happens at
// package init (and in a handful of tests); lookups happen on the per-node
// path of the mixed-radix recursion driver, hundreds of times per transform.
// The RWMutex this replaces cost 12% of runtime at n = 1000 (complex64) in
// atomic contention alone. Copy-on-write also makes the *CodeletEntry that
// Lookup returns stable: it points into a slice no later Register will sort
// or reallocate in place.
type CodeletRegistry[T fftypes.Complex] struct {
	mu       sync.Mutex // serializes writers only
	codelets atomic.Pointer[map[int][]CodeletEntry[T]]
}

// NewCodeletRegistry creates a new empty codelet registry.
func NewCodeletRegistry[T fftypes.Complex]() *CodeletRegistry[T] {
	r := &CodeletRegistry[T]{}
	empty := make(map[int][]CodeletEntry[T])
	r.codelets.Store(&empty)

	return r
}

// snapshot returns the current size map. The returned map must be treated as
// read-only: writers replace it wholesale rather than mutating it in place.
func (r *CodeletRegistry[T]) snapshot() map[int][]CodeletEntry[T] {
	m := r.codelets.Load()
	if m == nil {
		return nil
	}

	return *m
}

// Register adds a codelet to the registry.
// Multiple codelets can be registered for the same size (e.g., generic and SIMD variants).
// When looking up, the best available codelet for the current CPU is selected.
func (r *CodeletRegistry[T]) Register(entry CodeletEntry[T]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	current := r.snapshot()

	// Copy-on-write: build a fresh map and entry slice so concurrent readers
	// (and any *CodeletEntry they already hold) keep seeing a consistent,
	// immutable snapshot.
	updated := make(map[int][]CodeletEntry[T], len(current)+1)
	maps.Copy(updated, current)

	existing := current[entry.Size]
	entries := make([]CodeletEntry[T], len(existing), len(existing)+1)
	copy(entries, existing)
	entries = append(entries, entry)

	// Sort by rank level (higher = better) then priority
	sort.Slice(entries, func(i, j int) bool {
		li, lj := entries[i].rank(), entries[j].rank()
		if li != lj {
			return li > lj
		}

		return entries[i].Priority > entries[j].Priority
	})

	updated[entry.Size] = entries
	r.codelets.Store(&updated)
}

// Lookup finds the best codelet for a given size and CPU features.
// Returns nil if no codelet is available for the size.
// The lookup prefers higher SIMD levels that the CPU supports.
// Codelets with negative priority are skipped (disabled codelets).
func (r *CodeletRegistry[T]) Lookup(size int, features cpu.Features) *CodeletEntry[T] {
	return lookupIn(r.snapshot(), size, features)
}

// LookupBySignature finds a codelet by its signature.
// Used primarily for wisdom system lookups.
//
// Disabled codelets (negative priority) are skipped, matching Lookup. Wisdom
// entries are persisted by signature and can outlive a retuning that disables
// a codelet — or be imported from another machine — so without this filter a
// stale wisdom entry would resurrect a codelet that was deliberately measured
// to be slower than its alternatives.
func (r *CodeletRegistry[T]) LookupBySignature(size int, signature string) *CodeletEntry[T] {
	entries := r.snapshot()[size]
	for i := range entries {
		// Skip disabled codelets (negative priority)
		if entries[i].Priority < 0 {
			continue
		}

		if entries[i].Signature == signature {
			return &entries[i]
		}
	}

	return nil
}

// Sizes returns all sizes that have registered codelets.
func (r *CodeletRegistry[T]) Sizes() []int {
	current := r.snapshot()

	sizes := make([]int, 0, len(current))
	for size := range current {
		sizes = append(sizes, size)
	}

	return sizes
}

// Has returns true if there are any registered codelets for the given size.
func (r *CodeletRegistry[T]) Has(size int) bool {
	return len(r.snapshot()[size]) > 0
}

// GetAllForSize returns all registered codelets for a given size, regardless of CPU features.
// This is useful for testing all variants of a codelet.
func (r *CodeletRegistry[T]) GetAllForSize(size int) []CodeletEntry[T] {
	entries := r.snapshot()[size]
	if len(entries) == 0 {
		return nil
	}

	// Return a copy to avoid data races
	result := make([]CodeletEntry[T], len(entries))
	copy(result, entries)

	return result
}

// GetAvailableSizes returns all sizes with registered codelets that are
// compatible with the given CPU features. The returned slice is sorted in ascending order.
func (r *CodeletRegistry[T]) GetAvailableSizes(features cpu.Features) []int {
	current := r.snapshot()

	sizes := make([]int, 0, len(current))
	for size := range current {
		// Check if there's a codelet compatible with CPU features
		if lookupIn(current, size, features) != nil {
			sizes = append(sizes, size)
		}
	}

	sort.Ints(sizes)

	return sizes
}

// lookupIn resolves a codelet against an already-taken snapshot, so callers
// that scan many sizes take the snapshot once. It skips disabled codelets
// (negative priority) so that GetAvailableSizes never advertises a size that
// a subsequent Lookup would reject.
func lookupIn[T fftypes.Complex](codelets map[int][]CodeletEntry[T], size int, features cpu.Features) *CodeletEntry[T] {
	entries := codelets[size]
	if len(entries) == 0 {
		return nil
	}

	// Find the best codelet that the CPU supports
	for i := range entries {
		// Skip disabled codelets (negative priority)
		if entries[i].Priority < 0 {
			continue
		}

		if CPUSupports(features, entries[i].SIMDLevel) {
			return &entries[i]
		}
	}

	return nil
}

// CPUSupports checks if the CPU features support the given SIMD level.
func CPUSupports(features cpu.Features, level fftypes.SIMDLevel) bool {
	// ForceGeneric is a testing/debugging knob to disable *all* SIMD. It must
	// also apply to codelet selection; otherwise codelets can still pick SIMD
	// even when asm-dispatch selection is forced to generic.
	if features.ForceGeneric && level != fftypes.SIMDNone {
		return false
	}

	switch level {
	case fftypes.SIMDNone:
		return true
	case fftypes.SIMDSSE2:
		return features.HasSSE2
	case fftypes.SIMDSSE3:
		return features.HasSSE3
	case fftypes.SIMDAVX2:
		// The AVX2 codelet tier is uniformly FMA-dependent (its complex
		// multiplies compile to VFMADDSUB/VFMADD), so require HasFMA too.
		// FMA is a separate CPUID bit from AVX2: every real AVX2 CPU ships
		// FMA3, but emulators/VMs can mask it, and executing an FMA opcode
		// there faults. When FMA is absent we correctly fall back to the
		// SSE/generic tiers instead.
		return features.HasAVX2 && features.HasFMA
	case fftypes.SIMDAVX512:
		return features.HasAVX512
	case fftypes.SIMDNEON:
		return features.HasNEON
	default:
		return false
	}
}

// Global codelet registries, populated by internal/kernels at init time.
//
//nolint:gochecknoglobals
var (
	Registry64  = NewCodeletRegistry[complex64]()
	Registry128 = NewCodeletRegistry[complex128]()
)

// GetRegistry returns the appropriate registry for type T.
func GetRegistry[T fftypes.Complex]() *CodeletRegistry[T] {
	var zero T

	switch any(zero).(type) {
	case complex64:
		reg, ok := any(Registry64).(*CodeletRegistry[T])
		if !ok {
			panic("algofft: internal consistency error (Registry64)")
		}

		return reg
	case complex128:
		reg, ok := any(Registry128).(*CodeletRegistry[T])
		if !ok {
			panic("algofft: internal consistency error (Registry128)")
		}

		return reg
	default:
		return nil
	}
}
