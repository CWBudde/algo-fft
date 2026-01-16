package algofft

import "testing"

// TestDITScalingRoundTrip tests round-trip with DIT strategy.
//
//nolint:paralleltest // Modifies global kernel strategy state via SetKernelStrategy
func TestDITScalingRoundTrip(t *testing.T) {
	// NOT parallel - this test modifies global planner.kernelStrategy state
	prev := GetKernelStrategy()
	defer SetKernelStrategy(prev)

	SetKernelStrategy(KernelDIT)

	plan, err := NewPlanT[complex64](16)
	if err != nil {
		t.Fatalf("NewPlan(16) returned error: %v", err)
	}

	src := make([]complex64, 16)
	for i := range src {
		src[i] = complex(float32(i+1), float32(i))
	}

	freq := make([]complex64, 16)
	if err := plan.Forward(freq, src); err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	out := make([]complex64, 16)
	if err := plan.Inverse(out, freq); err != nil {
		t.Fatalf("Inverse() returned error: %v", err)
	}

	for i := range src {
		assertApproxComplex64f(t, out[i], src[i], 1e-4, "out[%d]", i)
	}
}

func TestOutOfPlaceDoesNotModifySource(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](16)
	if err != nil {
		t.Fatalf("NewPlan(16) returned error: %v", err)
	}

	src := make([]complex64, 16)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	orig := append([]complex64(nil), src...)

	dst := make([]complex64, 16)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	for i := range src {
		if src[i] != orig[i] {
			t.Fatalf("src[%d] modified: got %v want %v", i, src[i], orig[i])
		}
	}
}

func TestInPlaceMatchesOutOfPlace(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](32)
	if err != nil {
		t.Fatalf("NewPlan(32) returned error: %v", err)
	}

	src := make([]complex64, 32)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	out := make([]complex64, 32)
	if err := plan.Forward(out, src); err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	inplace := append([]complex64(nil), src...)
	if err := plan.InPlace(inplace); err != nil {
		t.Fatalf("InPlace() returned error: %v", err)
	}

	for i := range out {
		assertApproxComplex64f(t, inplace[i], out[i], 1e-4, "inplace[%d]", i)
	}
}

func TestTransformForwardInverse(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](16)
	if err != nil {
		t.Fatalf("NewPlan(16) returned error: %v", err)
	}

	src := make([]complex64, 16)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	freq := make([]complex64, 16)
	if err := plan.Transform(freq, src, false); err != nil {
		t.Fatalf("Transform forward returned error: %v", err)
	}

	out := make([]complex64, 16)
	if err := plan.Transform(out, freq, true); err != nil {
		t.Fatalf("Transform inverse returned error: %v", err)
	}

	for i := range src {
		assertApproxComplex64f(t, out[i], src[i], 1e-4, "out[%d]", i)
	}
}

func TestRoundTripSizes(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024}
	for _, n := range sizes {
		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		t.Logf("n=%d strategy=%v algorithm=%s", n, plan.kernelStrategy, plan.algorithm)

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i+1), float32(-i))
		}

		freq := make([]complex64, n)
		if err := plan.Forward(freq, src); err != nil {
			t.Fatalf("Forward(%d) returned error: %v", n, err)
		}

		out := make([]complex64, n)
		if err := plan.Inverse(out, freq); err != nil {
			t.Fatalf("Inverse(%d) returned error: %v", n, err)
		}

		for i := range src {
			assertApproxComplex64f(t, out[i], src[i], 1e-3, "n=%d out[%d]", n, i)
		}
	}
}

func TestRoundTripSizesComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32}
	for _, n := range sizes {
		plan, err := NewPlanT[complex128](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i+1), float64(-i))
		}

		freq := make([]complex128, n)
		if err := plan.Forward(freq, src); err != nil {
			t.Fatalf("Forward(%d) returned error: %v", n, err)
		}

		out := make([]complex128, n)
		if err := plan.Inverse(out, freq); err != nil {
			t.Fatalf("Inverse(%d) returned error: %v", n, err)
		}

		for i := range src {
			assertApproxComplex128f(t, out[i], src[i], 1e-10, "n=%d out[%d]", n, i)
		}
	}
}
