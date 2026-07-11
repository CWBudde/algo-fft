package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// Reset clears the scratch buffer and resets internal state.
// This can be useful to ensure deterministic behavior or to clear sensitive data.
// The Plan remains usable after Reset.
func (p *Plan[T]) Reset() {
	// Clear scratch buffer if it exists (pooled plans)
	if p.scratch != nil {
		clear(p.scratch)
	}

	if p.stridedScratch != nil {
		clear(p.stridedScratch)
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

	// Clear references to prevent reuse after Close
	p.pool = nil
	p.twiddle = nil
	p.codeletTwiddleForward = nil
	p.codeletTwiddleInverse = nil
	p.scratch = nil
	p.stridedScratch = nil
	p.bitrev = nil
	p.twiddleBacking = nil
	p.codeletTwiddleForwardBacking = nil
	p.codeletTwiddleInverseBacking = nil
	p.scratchBacking = nil
	p.stridedScratchBacking = nil
}

// Clone creates an independent copy of the Plan with its own scratch buffer.
// This is useful when multiple goroutines need to perform transforms concurrently,
// as each goroutine should use its own Plan to avoid data races on the scratch buffer.
//
// The cloned Plan shares immutable data (twiddle factors, bit-reversal indices)
// with the original for memory efficiency, but has its own scratch buffer.
//
// Cloned Plans are never pooled, even if the original was.
// Calling Close() on a cloned Plan is a no-op.
func (p *Plan[T]) Clone() *Plan[T] {
	scratchSize := p.n

	switch p.kernelStrategy {
	case fft.KernelBluestein:
		scratchSize = p.bluesteinM
	case fft.KernelRecursive:
		scratchSize = fft.ScratchSizeRecursive(p.decompStrategy)
	}

	scratch, scratchBacking := mem.AllocAligned[T](scratchSize)
	stridedScratch, stridedScratchBacking := mem.AllocAligned[T](p.n)

	var (
		bluesteinScratch        []T
		bluesteinScratchBacking []byte
	)

	if p.kernelStrategy == fft.KernelBluestein {
		bluesteinScratch, bluesteinScratchBacking = mem.AllocAligned[T](p.bluesteinM)
	}

	return &Plan[T]{
		n:                            p.n,
		twiddle:                      p.twiddle, // Shared (immutable)
		codeletTwiddleForward:        p.codeletTwiddleForward,
		codeletTwiddleInverse:        p.codeletTwiddleInverse,
		scratch:                      scratch,             // New allocation
		stridedScratch:               stridedScratch,      // New allocation
		bitrev:                       p.bitrev,            // Shared (immutable)
		packedTwiddle4:               p.packedTwiddle4,    // Shared (immutable)
		packedTwiddle4Inv:            p.packedTwiddle4Inv, // Shared (immutable)
		packedTwiddle8:               p.packedTwiddle8,    // Shared (immutable)
		packedTwiddle16:              p.packedTwiddle16,   // Shared (immutable)
		forwardCodelet:               p.forwardCodelet,    // Shared (function pointer)
		inverseCodelet:               p.inverseCodelet,    // Shared (function pointer)
		algorithm:                    p.algorithm,         // Shared (immutable string)
		forwardKernel:                p.forwardKernel,
		inverseKernel:                p.inverseKernel,
		kernelStrategy:               p.kernelStrategy,
		decompStrategy:               p.decompStrategy,
		meta:                         p.meta,
		twiddleBacking:               p.twiddleBacking, // Shared reference (keeps original alive)
		codeletTwiddleForwardBacking: p.codeletTwiddleForwardBacking,
		codeletTwiddleInverseBacking: p.codeletTwiddleInverseBacking,
		scratchBacking:               scratchBacking, // New allocation
		stridedScratchBacking:        stridedScratchBacking,
		pool:                         nil, // Clones are never pooled
		scratchPool:                  nil, // Clones have fixed scratch

		// Bluestein fields
		bluesteinM:              p.bluesteinM,
		bluesteinChirp:          p.bluesteinChirp,
		bluesteinChirpInv:       p.bluesteinChirpInv,
		bluesteinFilter:         p.bluesteinFilter,
		bluesteinFilterInv:      p.bluesteinFilterInv,
		bluesteinTwiddle:        p.bluesteinTwiddle,
		bluesteinBitrev:         p.bluesteinBitrev,
		bluesteinScratch:        bluesteinScratch,        // New allocation
		bluesteinScratchBacking: bluesteinScratchBacking, // New allocation
	}
}
