package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// Reset clears the scratch buffer and resets internal state.
// This can be useful to ensure deterministic behavior or to clear sensitive data.
// The Plan remains usable after Reset.
func (p *Plan[T]) Reset() {
	// Clear fixed scratch buffers if they exist (pooled/cloned plans)
	if p.scratch != nil {
		clear(p.scratch)
	}

	if p.stridedScratch != nil {
		clear(p.stridedScratch)
	}

	if p.subScratch != nil {
		clear(p.subScratch)
	}
	// For scratchPool, we don't clear as they are transient
}

// Close releases pooled resources back to the buffer pool.
// After Close, the Plan must not be used.
//
// Close is only necessary for Plans created with NewPlanPooled or
// NewPlanPooledWithOptions. For Plans created with NewPlan, Close is a no-op.
//
// It is safe to call Close multiple times; subsequent calls are no-ops.
func (p *Plan[T]) Close() {
	if p.pool == nil {
		return // Not a pooled plan
	}

	if p.twiddleBacking != nil {
		fft.PoolPut(p.pool, p.n, p.twiddle, p.twiddleBacking)
	}

	// Scratch may be allocated larger than p.n (see standardScratchSize), so it
	// was drawn from the pool bucket keyed on its own length; return it there.
	if p.scratchBacking != nil {
		fft.PoolPut(p.pool, len(p.scratch), p.scratch, p.scratchBacking)
	}

	if p.stridedScratchBacking != nil {
		fft.PoolPut(p.pool, len(p.stridedScratch), p.stridedScratch, p.stridedScratchBacking)
	}

	if p.bitrev != nil {
		p.pool.PutIntSlice(p.n, p.bitrev)
	}

	// Clear references to prevent reuse after Close. Dropping the executor
	// releases its tables for GC (pooled plans never share it with clones of
	// other plans; clones of this plan keep their own reference).
	p.pool = nil
	p.exec = nil
	p.forwardCodelet = nil
	p.inverseCodelet = nil
	p.codeletTwiddleForward = nil
	p.codeletTwiddleInverse = nil
	p.twiddle = nil
	p.scratch = nil
	p.stridedScratch = nil
	p.bitrev = nil
	p.twiddleBacking = nil
	p.scratchBacking = nil
	p.stridedScratchBacking = nil
}

// Clone creates an independent copy of the Plan with its own scratch buffers.
// This is useful when multiple goroutines need to perform transforms concurrently,
// as each goroutine should use its own Plan to avoid data races on the scratch buffer.
//
// The cloned Plan shares immutable data (the executor with its precomputed
// tables, twiddle factors, bit-reversal indices) with the original for memory
// efficiency, but has its own scratch buffers.
//
// Cloned Plans are never pooled, even if the original was.
// Calling Close() on a cloned Plan is a no-op.
func (p *Plan[T]) Clone() *Plan[T] {
	scratch, scratchBacking := mem.AllocAligned[T](p.scratchLen)
	stridedScratch, stridedScratchBacking := mem.AllocAligned[T](p.n)

	var (
		subScratch        []T
		subScratchBacking []byte
	)

	if p.subScratchLen > 0 {
		subScratch, subScratchBacking = mem.AllocAligned[T](p.subScratchLen)
	}

	return &Plan[T]{
		n:                p.n,
		exec:             p.exec, // Shared (immutable after construction)
		algorithm:        p.algorithm,
		forwardAlgorithm: p.forwardAlgorithm,
		inverseAlgorithm: p.inverseAlgorithm,
		kernelStrategy:   p.kernelStrategy,
		twiddle:          p.twiddle,        // Shared (immutable)
		bitrev:           p.bitrev,         // Shared (immutable)
		twiddleBacking:   p.twiddleBacking, // Shared reference (keeps original alive)

		// Shared codelet fast-path cache (function pointers + immutable tables)
		forwardCodelet:        p.forwardCodelet,
		inverseCodelet:        p.inverseCodelet,
		codeletTwiddleForward: p.codeletTwiddleForward,
		codeletTwiddleInverse: p.codeletTwiddleInverse,

		// Fresh scratch allocations
		scratch:               scratch,
		stridedScratch:        stridedScratch,
		subScratch:            subScratch,
		scratchBacking:        scratchBacking,
		stridedScratchBacking: stridedScratchBacking,
		subScratchBacking:     subScratchBacking,
		scratchLen:            p.scratchLen,
		subScratchLen:         p.subScratchLen,

		pool:        nil, // Clones are never pooled
		scratchPool: nil, // Clones have fixed scratch
	}
}
