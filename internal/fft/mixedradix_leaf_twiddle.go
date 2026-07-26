//go:build amd64 && !purego

package fft

import (
	"sync"

	m "github.com/cwbudde/algo-fft/internal/math"
)

// Leaf twiddle tables for the mixed-radix codelet dispatch.
//
// When the driver hands a size-n sub-transform to a codelet it must supply the
// standard size-n twiddle table. The obvious way to produce it — gathering
// twiddle[i*step] for i < n — costs n strided loads plus a pooled buffer on
// every call, and the result is the same table every time: the recursion keeps
// the invariant n*step == len(twiddle), so
//
//	twiddle[i*step] = W_len(twiddle)^(i*step) = W_n^i
//
// which is exactly ComputeTwiddleFactors[T](n). Caching it by size turns a
// per-call gather into a map lookup. The tables are immutable once published
// and shared across plans, matching how kernels' prepared-twiddle cache
// already works.
//
// Callers must verify the invariant (see leafTwiddleUsable) rather than assume
// it: a table longer than n*step would encode different roots of unity, and
// substituting the size-n table would silently transform the wrong thing.
//
//nolint:gochecknoglobals
var (
	leafTwiddleCache64  sync.Map // map[int][]complex64
	leafTwiddleCache128 sync.Map // map[int][]complex128
)

// leafTwiddleUsable reports whether the size-n standard twiddle table is
// interchangeable with a stride-step gather from a table of length tableLen.
func leafTwiddleUsable(n, step, tableLen int) bool {
	return n > 0 && step > 0 && n*step == tableLen
}

func leafTwiddle64(n int) []complex64 {
	if v, ok := leafTwiddleCache64.Load(n); ok {
		table, _ := v.([]complex64)

		return table
	}

	table := m.ComputeTwiddleFactors[complex64](n)
	actual, _ := leafTwiddleCache64.LoadOrStore(n, table)
	stored, _ := actual.([]complex64)

	return stored
}

func leafTwiddle128(n int) []complex128 {
	if v, ok := leafTwiddleCache128.Load(n); ok {
		table, _ := v.([]complex128)

		return table
	}

	table := m.ComputeTwiddleFactors[complex128](n)
	actual, _ := leafTwiddleCache128.LoadOrStore(n, table)
	stored, _ := actual.([]complex128)

	return stored
}
