//go:build !race

package algofft

import "testing"

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestPlanTransformsNoAllocsComplex64(t *testing.T) {
	const n = 1024

	plan, err := NewPlanT[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
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

	plan, err := NewPlanT[complex128](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
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

	plan, err := NewPlanT[complex128](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
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

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
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
