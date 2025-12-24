package fft

import (
	"testing"
)

// TestBufferPool_Complex64 tests the complex64 buffer pool.
func TestBufferPool_Complex64(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}

	const size = 256

	// Get a buffer from the pool
	buf1, backing1 := pool.GetComplex64(size)
	if len(buf1) != size {
		t.Errorf("GetComplex64(%d) returned buffer with length %d", size, len(buf1))
	}

	if cap(buf1) < size {
		t.Errorf("GetComplex64(%d) returned buffer with capacity %d", size, cap(buf1))
	}

	if backing1 == nil {
		t.Error("GetComplex64 returned nil backing buffer")
	}

	// Put it back
	pool.PutComplex64(size, buf1, backing1)

	// Get another buffer - should reuse
	buf2, backing2 := pool.GetComplex64(size)
	if len(buf2) != size {
		t.Errorf("GetComplex64(%d) second call returned buffer with length %d", size, len(buf2))
	}

	pool.PutComplex64(size, buf2, backing2)
}

// TestBufferPool_Complex128 tests the complex128 buffer pool.
func TestBufferPool_Complex128(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}

	const size = 512

	buf1, backing1 := pool.GetComplex128(size)
	if len(buf1) != size {
		t.Errorf("GetComplex128(%d) returned buffer with length %d", size, len(buf1))
	}

	if cap(buf1) < size {
		t.Errorf("GetComplex128(%d) returned buffer with capacity %d", size, cap(buf1))
	}

	if backing1 == nil {
		t.Error("GetComplex128 returned nil backing buffer")
	}

	pool.PutComplex128(size, buf1, backing1)

	buf2, backing2 := pool.GetComplex128(size)
	if len(buf2) != size {
		t.Errorf("GetComplex128(%d) second call returned buffer with length %d", size, len(buf2))
	}

	pool.PutComplex128(size, buf2, backing2)
}

// TestBufferPool_IntSlice tests the int slice buffer pool.
func TestBufferPool_IntSlice(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}

	const size = 128

	buf1 := pool.GetIntSlice(size)
	if len(buf1) != size {
		t.Errorf("GetIntSlice(%d) returned buffer with length %d", size, len(buf1))
	}

	if cap(buf1) < size {
		t.Errorf("GetIntSlice(%d) returned buffer with capacity %d", size, cap(buf1))
	}

	pool.PutIntSlice(size, buf1)

	buf2 := pool.GetIntSlice(size)
	if len(buf2) != size {
		t.Errorf("GetIntSlice(%d) second call returned buffer with length %d", size, len(buf2))
	}

	pool.PutIntSlice(size, buf2)
}

// TestBufferPool_MultipleSizes tests pooling with different sizes.
func TestBufferPool_MultipleSizes(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		// Get and return buffers
		buf64, backing64 := pool.GetComplex64(size)
		if len(buf64) != size {
			t.Errorf("size=%d: GetComplex64 returned length %d", size, len(buf64))
		}

		pool.PutComplex64(size, buf64, backing64)

		buf128, backing128 := pool.GetComplex128(size)
		if len(buf128) != size {
			t.Errorf("size=%d: GetComplex128 returned length %d", size, len(buf128))
		}

		pool.PutComplex128(size, buf128, backing128)

		bufInt := pool.GetIntSlice(size)
		if len(bufInt) != size {
			t.Errorf("size=%d: GetIntSlice returned length %d", size, len(bufInt))
		}

		pool.PutIntSlice(size, bufInt)
	}
}

// TestBufferPool_Concurrent tests concurrent pool access.
func TestBufferPool_Concurrent(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}

	const (
		size          = 256
		numGoroutines = 10
	)

	done := make(chan bool, numGoroutines)

	for range numGoroutines {
		go func() {
			// Get buffers
			buf64, backing64 := pool.GetComplex64(size)
			buf128, backing128 := pool.GetComplex128(size)
			bufInt := pool.GetIntSlice(size)

			// Verify lengths
			if len(buf64) != size || len(buf128) != size || len(bufInt) != size {
				t.Error("Buffer length mismatch in concurrent access")
			}

			// Return buffers
			pool.PutComplex64(size, buf64, backing64)
			pool.PutComplex128(size, buf128, backing128)
			pool.PutIntSlice(size, bufInt)

			done <- true
		}()
	}

	// Wait for all goroutines
	for range numGoroutines {
		<-done
	}
}

// TestBufferPool_WrongSizeIgnored tests that wrong-size buffers are not pooled.
func TestBufferPool_WrongSizeIgnored(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}

	// Get a buffer of size 64
	buf64, backing64 := pool.GetComplex64(64)

	// Try to return it as size 128 - should be ignored
	pool.PutComplex64(128, buf64, backing64)

	// Get a size 128 buffer - should be freshly allocated (not the size-64 buffer)
	buf128, backing128 := pool.GetComplex128(128)
	if len(buf128) != 128 {
		t.Errorf("Expected length 128, got %d", len(buf128))
	}

	pool.PutComplex128(128, buf128, backing128)
}

// TestBufferPool_ZeroSizeHandling tests edge case of zero-size requests.
func TestBufferPool_ZeroSizeHandling(t *testing.T) {
	t.Parallel()

	pool := &BufferPool{}

	// Getting a zero-size buffer should still work
	buf := pool.GetIntSlice(0)
	if buf == nil {
		t.Error("GetIntSlice(0) returned nil")
	}

	if len(buf) != 0 {
		t.Errorf("GetIntSlice(0) returned length %d, want 0", len(buf))
	}
}

// TestDefaultPool tests the global default pool.
func TestDefaultPool(t *testing.T) {
	t.Parallel()

	if DefaultPool == nil {
		t.Fatal("DefaultPool is nil")
	}

	// Verify it works
	buf, backing := DefaultPool.GetComplex64(128)
	if len(buf) != 128 {
		t.Errorf("DefaultPool.GetComplex64(128) returned length %d", len(buf))
	}

	DefaultPool.PutComplex64(128, buf, backing)
}

// BenchmarkBufferPool_GetPut benchmarks pool get/put operations.
func BenchmarkBufferPool_GetPut_Complex64(b *testing.B) {
	pool := &BufferPool{}
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(itoa(size), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				buf, backing := pool.GetComplex64(size)
				pool.PutComplex64(size, buf, backing)
			}
		})
	}
}

func BenchmarkBufferPool_GetPut_Complex128(b *testing.B) {
	pool := &BufferPool{}
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(itoa(size), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				buf, backing := pool.GetComplex128(size)
				pool.PutComplex128(size, buf, backing)
			}
		})
	}
}

func BenchmarkBufferPool_GetPut_IntSlice(b *testing.B) {
	pool := &BufferPool{}
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(itoa(size), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				buf := pool.GetIntSlice(size)
				pool.PutIntSlice(size, buf)
			}
		})
	}
}

// itoa is a simple int-to-string converter for benchmark names.
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
