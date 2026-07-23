//go:build !race

package algofft

import (
	"runtime"
	"strconv"
	"testing"
)

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanTransformsNoAllocsComplex64(t *testing.T) {
	const n = 1024

	plan, err := NewPlan[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan[complex64](%d) returned error: %v", n, err)
	}

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, n)
	freq := make([]complex64, n)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error {
		return plan.Forward(dst, src)
	})
	assertNoAllocs(t, "Inverse", func() error {
		return plan.Inverse(dst, freq)
	})
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanTransformsNoAllocsComplex128(t *testing.T) {
	const n = 1024

	plan, err := NewPlan[complex128](n)
	if err != nil {
		t.Fatalf("NewPlan[complex64](%d) returned error: %v", n, err)
	}

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, n)
	freq := make([]complex128, n)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error {
		return plan.Forward(dst, src)
	})
	assertNoAllocs(t, "Inverse", func() error {
		return plan.Inverse(dst, freq)
	})
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanTransformsNoAllocsComplex128_8192(t *testing.T) {
	const n = 8192

	plan, err := NewPlan[complex128](n)
	if err != nil {
		t.Fatalf("NewPlan[complex64](%d) returned error: %v", n, err)
	}

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, n)
	freq := make([]complex128, n)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	allocs := testing.AllocsPerRun(20, func() {
		err := plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward() returned error: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Forward allocated %.2f per run, want 0", allocs)
	}

	allocs = testing.AllocsPerRun(20, func() {
		err := plan.Inverse(dst, freq)
		if err != nil {
			t.Fatalf("Inverse() returned error: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Inverse allocated %.2f per run, want 0", allocs)
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanRealTransformsNoAllocs(t *testing.T) {
	const n = 1024

	plan, err := NewPlanReal[float32, complex64](n)
	if err != nil {
		t.Fatalf("NewPlanReal[float32, complex64](%d) returned error: %v", n, err)
	}

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 0.25
	}

	freq := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	out := make([]float32, n)

	assertNoAllocs(t, "Forward", func() error {
		return plan.Forward(freq, src)
	})
	assertNoAllocs(t, "Inverse", func() error {
		return plan.Inverse(out, freq)
	})
}

// Odd lengths run the full-size complex fallback; the guard covers both the
// widen/slice forward path and the Hermitian-rebuild inverse path.
//
//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanRealOddTransformsNoAllocs(t *testing.T) {
	const n = 105

	plan, err := NewPlanReal[float32, complex64](n)
	if err != nil {
		t.Fatalf("NewPlanReal[float32, complex64](%d) returned error: %v", n, err)
	}

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 0.25
	}

	freq := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	out := make([]float32, n)

	assertNoAllocs(t, "Forward", func() error {
		return plan.Forward(freq, src)
	})
	assertNoAllocs(t, "Inverse", func() error {
		return plan.Inverse(out, freq)
	})
}

func assertNoAllocs(t *testing.T, label string, run func() error) {
	t.Helper()

	allocs := testing.AllocsPerRun(100, func() {
		err := run()
		if err != nil {
			t.Fatalf("%s returned error: %v", label, err)
		}
	})

	if allocs != 0 {
		t.Fatalf("%s allocated %.2f per run, want 0", label, allocs)
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlan2DTransformsNoAllocsComplex64(t *testing.T) {
	const rows, cols = 16, 16

	plan, err := NewPlan2D32(rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D32(%d, %d) returned error: %v", rows, cols, err)
	}

	src := make([]complex64, rows*cols)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, rows*cols)
	freq := make([]complex64, rows*cols)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
	assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlan2DTransformsNoAllocsComplex128(t *testing.T) {
	const rows, cols = 16, 16

	plan, err := NewPlan2D64(rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D64(%d, %d) returned error: %v", rows, cols, err)
	}

	src := make([]complex128, rows*cols)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, rows*cols)
	freq := make([]complex128, rows*cols)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
	assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlan3DTransformsNoAllocsComplex64(t *testing.T) {
	const depth, height, width = 8, 8, 8

	plan, err := NewPlan3D32(depth, height, width)
	if err != nil {
		t.Fatalf("NewPlan3D32(%d, %d, %d) returned error: %v", depth, height, width, err)
	}

	src := make([]complex64, depth*height*width)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, depth*height*width)
	freq := make([]complex64, depth*height*width)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
	assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlan3DTransformsNoAllocsComplex128(t *testing.T) {
	const depth, height, width = 8, 8, 8

	plan, err := NewPlan3D64(depth, height, width)
	if err != nil {
		t.Fatalf("NewPlan3D64(%d, %d, %d) returned error: %v", depth, height, width, err)
	}

	src := make([]complex128, depth*height*width)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, depth*height*width)
	freq := make([]complex128, depth*height*width)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
	assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanNDTransformsNoAllocsComplex64(t *testing.T) {
	dims := []int{8, 8, 8}

	plan, err := NewPlanND32(dims)
	if err != nil {
		t.Fatalf("NewPlanND32(%v) returned error: %v", dims, err)
	}

	src := make([]complex64, plan.Len())
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, plan.Len())
	freq := make([]complex64, plan.Len())

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
	assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanNDTransformsNoAllocsComplex128(t *testing.T) {
	dims := []int{4, 4, 4, 4}

	plan, err := NewPlanND64(dims)
	if err != nil {
		t.Fatalf("NewPlanND64(%v) returned error: %v", dims, err)
	}

	src := make([]complex128, plan.Len())
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, plan.Len())
	freq := make([]complex128, plan.Len())

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
	assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
}

// TestPlanMixedRadixTransformsNoAllocs guards the zero-allocation promise on the
// mixed-radix (highly-composite, non-power-of-2) path. These sizes exercise the
// pooled radix schedule buffer and, on SIMD builds, the pooled sub-transform
// twiddle/scratch buffers and the size-384 codelet's pooled internals.
//
//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanMixedRadixTransformsNoAllocs(t *testing.T) {
	// 768 = 2^8·3, 1536 = 2^9·3 (routes through the size-384 codelet under asm).
	for _, n := range []int{96, 768, 1536} {
		t.Run("complex64_"+strconv.Itoa(n), func(t *testing.T) {
			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) returned error: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i+1), float32(-i))
			}

			dst := make([]complex64, n)
			freq := make([]complex64, n)

			err = plan.Forward(freq, src)
			if err != nil {
				t.Fatalf("Forward() returned error: %v", err)
			}

			assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
			assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
		})

		t.Run("complex128_"+strconv.Itoa(n), func(t *testing.T) {
			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex128](%d) returned error: %v", n, err)
			}

			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i+1), float64(-i))
			}

			dst := make([]complex128, n)
			freq := make([]complex128, n)

			err = plan.Forward(freq, src)
			if err != nil {
				t.Fatalf("Forward() returned error: %v", err)
			}

			assertNoAllocs(t, "Forward", func() error { return plan.Forward(dst, src) })
			assertNoAllocs(t, "Inverse", func() error { return plan.Inverse(dst, freq) })
		})
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanReal2DTransformsNoAllocs(t *testing.T) {
	const rows, cols = 16, 16

	plan, err := NewPlanReal2D[float32, complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlanReal2D[float32, complex64](%d, %d) returned error: %v", rows, cols, err)
	}

	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(i) * 0.5
	}

	compact := make([]complex64, rows*(cols/2+1))
	full := make([]complex64, rows*cols)

	// Warm the resident scratch slot so the first measured run is steady-state.
	err = plan.Forward(compact, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	err = plan.ForwardFull(full, src)
	if err != nil {
		t.Fatalf("ForwardFull() returned error: %v", err)
	}

	// Force a GC each iteration to drain the overflow pool: this catches
	// ForwardFull nesting a second scratch borrow through Forward, which would
	// re-allocate once the pool is empty.
	assertNoAllocsGC(t, "Forward", func() error { return plan.Forward(compact, src) })
	assertNoAllocsGC(t, "ForwardFull", func() error { return plan.ForwardFull(full, src) })
}

// TestBluesteinTransformsNoAllocs locks in zero allocations for Bluestein
// plans: n=509 pads to m=1024 (size-dispatched DIT sub-FFT), n=4099 pads to
// m=16384 (above the dispatch bound, generic radix-2 with the plan's cached
// bit-reversal table).
//
//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestBluesteinTransformsNoAllocs(t *testing.T) {
	for _, n := range []int{509, 4099} {
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) returned error: %v", n, err)
		}

		if plan.KernelStrategy() != KernelBluestein {
			t.Fatalf("NewPlan[complex64](%d) strategy = %v, want KernelBluestein", n, plan.KernelStrategy())
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i+1), float32(-i))
		}

		dst := make([]complex64, n)
		freq := make([]complex64, n)

		// Warm the resident scratch slot so the first measured run is steady-state.
		err = plan.Forward(freq, src)
		if err != nil {
			t.Fatalf("Forward() returned error: %v", err)
		}

		assertNoAllocs(t, "Forward", func() error {
			return plan.Forward(dst, src)
		})
		assertNoAllocs(t, "Inverse", func() error {
			return plan.Inverse(dst, freq)
		})
	}
}

// assertNoAllocsGC is assertNoAllocs but forces a GC before each run so that
// buffers cached opportunistically in the overflow sync.Pool are reclaimed,
// exercising the resident-only allocation-free path.
func assertNoAllocsGC(t *testing.T, label string, run func() error) {
	t.Helper()

	allocs := testing.AllocsPerRun(50, func() {
		runtime.GC()

		err := run()
		if err != nil {
			t.Fatalf("%s returned error: %v", label, err)
		}
	})

	if allocs != 0 {
		t.Fatalf("%s allocated %.2f per run, want 0", label, allocs)
	}
}
