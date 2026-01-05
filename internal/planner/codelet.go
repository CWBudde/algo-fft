package planner

import (
	"sort"
	"sync"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/fftypes"
)

// CodeletFunc is a type alias for the codelet function signature.
// The canonical definition is in internal/fftypes.
type CodeletFunc[T Complex] = fftypes.CodeletFunc[T]

// SIMDLevel is a type alias for SIMD feature levels.
// The canonical definition is in internal/fftypes.
type SIMDLevel = fftypes.SIMDLevel

// SIMD level constants - aliases for backward compatibility.
const (
	SIMDNone   = fftypes.SIMDNone
	SIMDSSE2   = fftypes.SIMDSSE2
	SIMDAVX2   = fftypes.SIMDAVX2
	SIMDAVX512 = fftypes.SIMDAVX512
	SIMDNEON   = fftypes.SIMDNEON
)

// BitrevFunc is a type alias for bit-reversal function signature.
// The canonical definition is in internal/fftypes.
type BitrevFunc = fftypes.BitrevFunc

// CodeletEntry describes a registered codelet for a specific size.
type CodeletEntry[T Complex] struct {
	Size       int            // FFT size this codelet handles
	Forward    CodeletFunc[T] // Forward transform (nil if not available)
	Inverse    CodeletFunc[T] // Inverse transform (nil if not available)
	Algorithm  KernelStrategy // DIT, Stockham, etc.
	SIMDLevel  SIMDLevel      // Required CPU features
	Signature  string         // Human-readable name: "dit8_avx2"
	Priority   int            // Higher priority = preferred (for same SIMD level)
	BitrevFunc BitrevFunc     // Bit-reversal generator (nil = no bit-reversal needed)
}

// CodeletRegistry provides size-indexed codelet lookup.
// Codelets are organized by size, with multiple implementations per size
// (e.g., generic, AVX2, NEON variants).
type CodeletRegistry[T Complex] struct {
	mu       sync.RWMutex
	codelets map[int][]CodeletEntry[T] // size -> codelets (sorted by preference)
}

// NewCodeletRegistry creates a new empty codelet registry.
func NewCodeletRegistry[T Complex]() *CodeletRegistry[T] {
	return &CodeletRegistry[T]{
		codelets: make(map[int][]CodeletEntry[T]),
	}
}

// Register adds a codelet to the registry.
// Multiple codelets can be registered for the same size (e.g., generic and SIMD variants).
// When looking up, the best available codelet for the current CPU is selected.
func (r *CodeletRegistry[T]) Register(entry CodeletEntry[T]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entries := r.codelets[entry.Size]
	entries = append(entries, entry)

	// Sort by SIMD level (higher = better) then priority
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].SIMDLevel != entries[j].SIMDLevel {
			return entries[i].SIMDLevel > entries[j].SIMDLevel
		}

		return entries[i].Priority > entries[j].Priority
	})

	r.codelets[entry.Size] = entries
}

// Lookup finds the best codelet for a given size and CPU features.
// Returns nil if no codelet is available for the size.
// The lookup prefers higher SIMD levels that the CPU supports.
// Codelets with negative priority are skipped (disabled codelets).
func (r *CodeletRegistry[T]) Lookup(size int, features cpu.Features) *CodeletEntry[T] {
	r.mu.RLock()
	defer r.mu.RUnlock()

	entries := r.codelets[size]
	if len(entries) == 0 {
		return nil
	}

	// Find the best codelet that the CPU supports
	for i := range entries {
		// Skip disabled codelets (negative priority)
		if entries[i].Priority < 0 {
			continue
		}

		if cpuSupports(features, entries[i].SIMDLevel) {
			return &entries[i]
		}
	}

	return nil
}

// LookupBySignature finds a codelet by its signature.
// Used primarily for wisdom system lookups.
func (r *CodeletRegistry[T]) LookupBySignature(size int, signature string) *CodeletEntry[T] {
	r.mu.RLock()
	defer r.mu.RUnlock()

	entries := r.codelets[size]
	for i := range entries {
		if entries[i].Signature == signature {
			return &entries[i]
		}
	}

	return nil
}

// Sizes returns all sizes that have registered codelets.
func (r *CodeletRegistry[T]) Sizes() []int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sizes := make([]int, 0, len(r.codelets))
	for size := range r.codelets {
		sizes = append(sizes, size)
	}

	return sizes
}

// Has returns true if there are any registered codelets for the given size.
func (r *CodeletRegistry[T]) Has(size int) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.codelets[size]) > 0
}

// GetAllForSize returns all registered codelets for a given size, regardless of CPU features.
// This is useful for testing all variants of a codelet.
func (r *CodeletRegistry[T]) GetAllForSize(size int) []CodeletEntry[T] {
	r.mu.RLock()
	defer r.mu.RUnlock()

	entries := r.codelets[size]
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
	r.mu.RLock()
	defer r.mu.RUnlock()

	sizes := make([]int, 0, len(r.codelets))
	for size := range r.codelets {
		// Check if there's a codelet compatible with CPU features
		if r.lookupUnlocked(size, features) != nil {
			sizes = append(sizes, size)
		}
	}

	sort.Ints(sizes)

	return sizes
}

// lookupUnlocked is an internal version of Lookup without locking.
// Caller must hold r.mu (read or write lock).
func (r *CodeletRegistry[T]) lookupUnlocked(size int, features cpu.Features) *CodeletEntry[T] {
	entries := r.codelets[size]
	if len(entries) == 0 {
		return nil
	}

	// Find the best codelet that the CPU supports
	for i := range entries {
		if cpuSupports(features, entries[i].SIMDLevel) {
			return &entries[i]
		}
	}

	return nil
}

// cpuSupports checks if the CPU features support the given SIMD level.
func cpuSupports(features cpu.Features, level SIMDLevel) bool {
	// ForceGeneric is a testing/debugging knob to disable *all* SIMD. It must
	// also apply to codelet selection; otherwise codelets can still pick SIMD
	// even when asm-dispatch selection is forced to generic.
	if features.ForceGeneric && level != SIMDNone {
		return false
	}

	switch level {
	case SIMDNone:
		return true
	case SIMDSSE2:
		return features.HasSSE2
	case SIMDAVX2:
		return features.HasAVX2
	case SIMDAVX512:
		return features.HasAVX512
	case SIMDNEON:
		return features.HasNEON
	default:
		return false
	}
}

// Global codelet registries, populated at init time.
//
//nolint:gochecknoglobals
var (
	Registry64  = NewCodeletRegistry[complex64]()
	Registry128 = NewCodeletRegistry[complex128]()
)

// GetRegistry returns the appropriate registry for type T.
func GetRegistry[T Complex]() *CodeletRegistry[T] {
	var zero T

	switch any(zero).(type) {
	case complex64:
		return any(Registry64).(*CodeletRegistry[T])
	case complex128:
		return any(Registry128).(*CodeletRegistry[T])
	default:
		return nil
	}
}
