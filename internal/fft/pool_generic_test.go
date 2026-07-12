package fft

import (
	"testing"
)

// TestPoolGetPut exercises the generic PoolGet/PoolPut dispatch for both
// precisions: buffers come back with the requested length and can be returned
// and reused without error.
func TestPoolGetPut(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()
		testPoolGetPut[complex64](t)
	})
	t.Run("complex128", func(t *testing.T) {
		t.Parallel()
		testPoolGetPut[complex128](t)
	})
}

func testPoolGetPut[T Complex](t *testing.T) {
	t.Helper()

	pool := &BufferPool{}

	const n = 64

	data, backing := PoolGet[T](pool, n)
	if len(data) != n {
		t.Fatalf("PoolGet returned len %d, want %d", len(data), n)
	}

	for i := range data {
		data[i] = T(complex(float64(i), -float64(i)))
	}

	PoolPut(pool, n, data, backing)

	// Nil data must be accepted silently.
	PoolPut[T](pool, n, nil, nil)

	again, backing2 := PoolGet[T](pool, n)
	if len(again) != n {
		t.Fatalf("second PoolGet returned len %d, want %d", len(again), n)
	}

	PoolPut(pool, n, again, backing2)
}

// TestComplexMulArrayInPlaceGeneric verifies the generic in-place multiply
// dispatch for both precisions against direct multiplication.
func TestComplexMulArrayInPlaceGeneric(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()
		testComplexMulArrayInPlace[complex64](t)
	})
	t.Run("complex128", func(t *testing.T) {
		t.Parallel()
		testComplexMulArrayInPlace[complex128](t)
	})
}

func testComplexMulArrayInPlace[T Complex](t *testing.T) {
	t.Helper()

	const n = 37 // odd length exercises SIMD tail handling

	dst := make([]T, n)
	src := make([]T, n)
	want := make([]T, n)

	for i := range dst {
		dst[i] = T(complex(float64(i+1), float64(-i)))
		src[i] = T(complex(0.5, float64(i)*0.25))
		want[i] = dst[i] * src[i]
	}

	ComplexMulArrayInPlace(dst, src)

	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("index %d: got %v, want %v", i, dst[i], want[i])
		}
	}
}

// TestScaleInPlace verifies scaling for both precisions, including the
// scale==1 fast path.
func TestScaleInPlace(t *testing.T) {
	t.Parallel()

	const n = 33

	src64 := randomComplex64(n, 0x5CA1E)

	dst64 := make([]complex64, n)
	copy(dst64, src64)
	ScaleComplex64InPlace(dst64, 1) // no-op fast path

	for i := range dst64 {
		if dst64[i] != src64[i] {
			t.Fatalf("scale=1 modified index %d", i)
		}
	}

	ScaleComplex64InPlace(dst64, 0.5)

	for i := range dst64 {
		want := src64[i] * 0.5
		if dst64[i] != want {
			t.Fatalf("complex64 index %d: got %v, want %v", i, dst64[i], want)
		}
	}

	src128 := randomComplex128(n, 0x5CA1F)

	dst128 := make([]complex128, n)
	copy(dst128, src128)
	ScaleComplex128InPlace(dst128, 1) // no-op fast path

	for i := range dst128 {
		if dst128[i] != src128[i] {
			t.Fatalf("scale=1 modified index %d", i)
		}
	}

	ScaleComplex128InPlace(dst128, 0.25)

	for i := range dst128 {
		want := src128[i] * 0.25
		if dst128[i] != want {
			t.Fatalf("complex128 index %d: got %v, want %v", i, dst128[i], want)
		}
	}
}
