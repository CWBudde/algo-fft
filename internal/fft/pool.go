package fft

import (
	"sync"

	mem "github.com/MeKo-Christian/algo-fft/internal/memory"
)

// BufferPool provides pooled allocations for FFT buffers to reduce GC pressure
// when creating and destroying many Plans with the same size.
//
// Buffers are organized by size class (power of 2) for efficient reuse.
// The pool automatically handles both complex64 and complex128 buffers.
type BufferPool struct {
	// pools64 maps FFT size → pool of complex64 aligned buffers
	pools64 sync.Map // map[int]*sync.Pool

	// pools128 maps FFT size → pool of complex128 aligned buffers
	pools128 sync.Map // map[int]*sync.Pool

	// poolsInt maps FFT size → pool of int slices (for bit-reversal indices)
	poolsInt sync.Map // map[int]*sync.Pool
}

// alignedBuffer64 holds an aligned complex64 buffer and its backing storage.
type alignedBuffer64 struct {
	data    []complex64
	backing []byte
}

// alignedBuffer128 holds an aligned complex128 buffer and its backing storage.
type alignedBuffer128 struct {
	data    []complex128
	backing []byte
}

// intBuffer holds an int slice for pooling.
type intBuffer struct {
	data []int
}

// DefaultPool is the global buffer pool used by NewPlanPooled.
var DefaultPool = &BufferPool{}

// GetComplex64 retrieves or allocates an aligned complex64 buffer of size n.
func (p *BufferPool) GetComplex64(n int) ([]complex64, []byte) {
	pool := p.getOrCreatePool64(n)

	buf, ok := pool.Get().(*alignedBuffer64)
	if !ok {
		panic("algofft: internal pool type error (complex64)")
	}

	return buf.data, buf.backing
}

// PutComplex64 returns a complex64 buffer to the pool for reuse.
// The buffer must have been obtained from GetComplex64 with the same size.
func (p *BufferPool) PutComplex64(n int, data []complex64, backing []byte) {
	if len(data) != n {
		return // Wrong size, don't pool
	}

	pool := p.getOrCreatePool64(n)
	pool.Put(&alignedBuffer64{data: data, backing: backing})
}

// GetComplex128 retrieves or allocates an aligned complex128 buffer of size n.
func (p *BufferPool) GetComplex128(n int) ([]complex128, []byte) {
	pool := p.getOrCreatePool128(n)

	buf, ok := pool.Get().(*alignedBuffer128)
	if !ok {
		panic("algofft: internal pool type error (complex128)")
	}

	return buf.data, buf.backing
}

// PutComplex128 returns a complex128 buffer to the pool for reuse.
// The buffer must have been obtained from GetComplex128 with the same size.
func (p *BufferPool) PutComplex128(n int, data []complex128, backing []byte) {
	if len(data) != n {
		return // Wrong size, don't pool
	}

	pool := p.getOrCreatePool128(n)
	pool.Put(&alignedBuffer128{data: data, backing: backing})
}

// GetIntSlice retrieves or allocates an int slice of size n.
func (p *BufferPool) GetIntSlice(n int) []int {
	pool := p.getOrCreatePoolInt(n)

	buf, ok := pool.Get().(*intBuffer)
	if !ok {
		panic("algofft: internal pool type error (int)")
	}

	return buf.data
}

// PutIntSlice returns an int slice to the pool for reuse.
func (p *BufferPool) PutIntSlice(n int, data []int) {
	if len(data) != n {
		return // Wrong size, don't pool
	}

	pool := p.getOrCreatePoolInt(n)
	pool.Put(&intBuffer{data: data})
}

func (p *BufferPool) getOrCreatePool64(n int) *sync.Pool {
	if existing, ok := p.pools64.Load(n); ok {
		pool, ok := existing.(*sync.Pool)
		if !ok {
			panic("algofft: internal pool map error (complex64)")
		}

		return pool
	}

	pool := &sync.Pool{
		New: func() any {
			data, backing := mem.AllocAlignedComplex64(n)
			return &alignedBuffer64{data: data, backing: backing}
		},
	}

	actual, _ := p.pools64.LoadOrStore(n, pool)

	apool, ok := actual.(*sync.Pool)
	if !ok {
		panic("algofft: internal pool map error (complex64)")
	}

	return apool
}

func (p *BufferPool) getOrCreatePool128(n int) *sync.Pool {
	if existing, ok := p.pools128.Load(n); ok {
		pool, ok := existing.(*sync.Pool)
		if !ok {
			panic("algofft: internal pool map error (complex128)")
		}

		return pool
	}

	pool := &sync.Pool{
		New: func() any {
			data, backing := mem.AllocAlignedComplex128(n)
			return &alignedBuffer128{data: data, backing: backing}
		},
	}

	actual, _ := p.pools128.LoadOrStore(n, pool)

	apool, ok := actual.(*sync.Pool)
	if !ok {
		panic("algofft: internal pool map error (complex128)")
	}

	return apool
}

func (p *BufferPool) getOrCreatePoolInt(n int) *sync.Pool {
	if existing, ok := p.poolsInt.Load(n); ok {
		pool, ok := existing.(*sync.Pool)
		if !ok {
			panic("algofft: internal pool map error (int)")
		}

		return pool
	}

	pool := &sync.Pool{
		New: func() any {
			return &intBuffer{data: make([]int, n)}
		},
	}

	actual, _ := p.poolsInt.LoadOrStore(n, pool)

	apool, ok := actual.(*sync.Pool)
	if !ok {
		panic("algofft: internal pool map error (int)")
	}

	return apool
}
