//go:build amd64 && asm && !purego

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
var (
	bitrevCacheMu sync.RWMutex
	bitrevCache   = map[int][]int{}
)

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
