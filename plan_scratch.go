package algofft

import (
	"sync"
	"sync/atomic"

	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// residentCache hands out per-call scratch sets so a single plan instance can
// run transforms from multiple goroutines concurrently. One set lives in a
// GC-proof resident slot so a plan used from a single goroutine never
// re-allocates: a bare sync.Pool is drained every two GC cycles, which would
// re-run the aligned scratch allocations on small heaps. Concurrent transform
// bursts overflow into the sync.Pool, so extras are cached opportunistically
// and released by the GC when the burst is over.
type residentCache[S any] struct {
	resident atomic.Pointer[S]
	pool     sync.Pool
}

// newResidentCache creates a cache whose sets are produced by alloc and seeds
// the resident slot so the first transform is allocation-free.
func newResidentCache[S any](alloc func() *S) *residentCache[S] {
	c := new(residentCache[S])
	c.pool.New = func() any {
		return alloc()
	}
	c.put(alloc())

	return c
}

func (c *residentCache[S]) get() *S {
	if s := c.resident.Swap(nil); s != nil {
		return s
	}

	s, ok := c.pool.Get().(*S)
	if !ok {
		panic("algofft: internal pool type error (scratch set)")
	}

	return s
}

func (c *residentCache[S]) put(s *S) {
	if c.resident.CompareAndSwap(nil, s) {
		return
	}

	c.pool.Put(s)
}

// allocAlignedSlice allocates a SIMD-aligned slice for the supported complex
// types, returning the slice and its backing array (kept alive for the GC).
func allocAlignedSlice[T Complex](n int) ([]T, []byte) {
	var zero T

	switch any(zero).(type) {
	case complex64:
		s, b := mem.AllocAlignedComplex64(n)

		t, ok := any(s).([]T)
		if !ok {
			panic("algofft: internal type error (aligned alloc)")
		}

		return t, b
	case complex128:
		s, b := mem.AllocAlignedComplex128(n)

		t, ok := any(s).([]T)
		if !ok {
			panic("algofft: internal type error (aligned alloc)")
		}

		return t, b
	default:
		return make([]T, n), nil
	}
}
