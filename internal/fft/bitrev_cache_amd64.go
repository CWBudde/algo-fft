//go:build amd64 && !purego

package fft

import (
	"sync"

	m "github.com/cwbudde/algo-fft/internal/math"
)

// bitrevCache memoizes radix-2 bit-reversal index tables keyed by transform
// size. The tables are read-only permutation maps — the AVX2/SSE assembly
// kernels only index through them, never write (see wrapAsmDIT64 and the static
// sse2_bitrev_tables, which already share one table per size across every
// call) — so a single cached slice per size is safe to hand to every transform.
//
// This keeps the AVX2 Stockham and complex128 wrappers allocation-free after the
// first transform of a given size, instead of rebuilding the table on every
// call. It is a pure size→indices memoization table (like the precomputed
// package-level tables), not mutable planner state, so it does not reintroduce
// the process-global tuning state removed in P1.1.
//
// Memory tradeoff: the cache is process-global and never evicted, so it keeps
// one table per distinct transform size for the life of the process. Growth is
// bounded by the number of distinct sizes actually transformed (typically a
// handful) and each table is small (8·n bytes), so this is an intentional
// space-for-speed choice — the same unbounded-by-size model the pre-existing
// prepared-twiddle cache uses — and it only exists on SIMD builds. A size
// cap/eviction would add branching and locking to a hot path for no practical
// benefit; if a future workload plans pathologically many distinct sizes, that
// is where a bound (or a per-plan cache) should be introduced.
var (
	bitrevCacheMu sync.RWMutex
	bitrevCache   = map[int][]int{}
)

// PrewarmSizeCaches populates the process-global per-size tables used by the
// SIMD kernel wrappers (currently the radix-2 bit-reversal table shared by
// the AVX-512 generic, AVX2 Stockham, and AVX2 complex128 wrappers) so that a
// plan's first transform does not allocate. Called at plan creation; a no-op
// for sizes the wrappers never handle (non-powers of two) and on non-SIMD
// builds (see bitrev_cache_stub.go).
func PrewarmSizeCaches(n int) {
	if n > 0 && m.IsPowerOf2(n) {
		cachedBitReversalIndices(n)
	}
}

// cachedBitReversalIndices returns the shared radix-2 bit-reversal table for
// size n, computing and caching it on first use.
func cachedBitReversalIndices(n int) []int {
	bitrevCacheMu.RLock()
	idx, ok := bitrevCache[n]
	bitrevCacheMu.RUnlock()

	if ok {
		return idx
	}

	idx = m.ComputeBitReversalIndices(n)

	bitrevCacheMu.Lock()
	// A concurrent caller may have populated the entry in the meantime; prefer
	// the existing slice so every caller shares exactly one table per size.
	if existing, ok := bitrevCache[n]; ok {
		idx = existing
	} else {
		bitrevCache[n] = idx
	}
	bitrevCacheMu.Unlock()

	return idx
}
