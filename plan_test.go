package algofft

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func TestNewPlan_PowersOfTwo(t *testing.T) {
	t.Parallel()

	sizes := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
	for _, n := range sizes {
		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Errorf("NewPlan(%d) returned error: %v", n, err)
			continue
		}

		if plan.Len() != n {
			t.Errorf("NewPlan(%d).Len() = %d, want %d", n, plan.Len(), n)
		}

		// Verify Forward works
		src := make([]complex64, n)

		dst := make([]complex64, n)
		if err := plan.Forward(dst, src); err != nil {
			t.Errorf("Forward(%d) returned error: %v", n, err)
		}
	}
}

func TestNewPlan32(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(256)
	if err != nil {
		t.Fatalf("NewPlan32(256) returned error: %v", err)
	}

	if plan.Len() != 256 {
		t.Errorf("NewPlan32(256).Len() = %d, want 256", plan.Len())
	}
}

func TestNewPlan64(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan64(256)
	if err != nil {
		t.Fatalf("NewPlan64(256) returned error: %v", err)
	}

	if plan.Len() != 256 {
		t.Errorf("NewPlan64(256).Len() = %d, want 256", plan.Len())
	}
}

func TestNewPlan_InvalidLength(t *testing.T) {
	t.Parallel()

	invalidSizes := []int{0, -1, -100}
	for _, n := range invalidSizes {
		plan, err := NewPlanT[complex64](n)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("NewPlan(%d) = (%v, %v), want (nil, ErrInvalidLength)", n, plan, err)
		}
	}
}

func TestNewPlan_MixedRadixLengths(t *testing.T) {
	t.Parallel()

	for _, n := range []int{6, 10, 12, 15, 20, 30, 60} {
		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		src[0] = 1

		dst := make([]complex64, n)
		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward(%d) returned error: %v", n, err)
		}

		for i := range dst {
			if absComplex64(dst[i]-1) > 1e-3 {
				t.Fatalf("n=%d dst[%d] = %v, want ~1", n, i, dst[i])
			}
		}
	}
}

func TestNewPlan_TwiddleFactors(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex128](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	// W_n^0 should be 1
	if cmplx.Abs(plan.twiddle[0]-1) > 1e-10 {
		t.Errorf("twiddle[0] = %v, want 1+0i", plan.twiddle[0])
	}

	// W_n^(n/2) should be -1 for even n
	n := plan.Len()
	expected := complex(-1, 0)

	if cmplx.Abs(plan.twiddle[n/2]-expected) > 1e-10 {
		t.Errorf("twiddle[%d] = %v, want -1+0i", n/2, plan.twiddle[n/2])
	}

	// W_n^(n/4) should be -i for n divisible by 4
	expectedQuarter := complex(0, -1)

	if cmplx.Abs(plan.twiddle[n/4]-expectedQuarter) > 1e-10 {
		t.Errorf("twiddle[%d] = %v, want 0-1i", n/4, plan.twiddle[n/4])
	}
}

func TestNewPlan_TwiddleFactorsPeriodicity(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex128](16)
	if err != nil {
		t.Fatalf("NewPlan(16) returned error: %v", err)
	}

	// Verify W_n^k has magnitude 1 (lies on unit circle)
	for k, w := range plan.twiddle {
		mag := cmplx.Abs(w)
		if math.Abs(mag-1) > 1e-10 {
			t.Errorf("twiddle[%d] has magnitude %v, want 1", k, mag)
		}
	}
}

// Tests for Forward/Inverse methods and error handling

func TestForward_NilSlice(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	// nil dst
	err = plan.Forward(nil, make([]complex64, 8))
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("Forward(nil, src) = %v, want ErrNilSlice", err)
	}

	// nil src
	err = plan.Forward(make([]complex64, 8), nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("Forward(dst, nil) = %v, want ErrNilSlice", err)
	}

	// both nil
	err = plan.Forward(nil, nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("Forward(nil, nil) = %v, want ErrNilSlice", err)
	}
}

func TestForward_LengthMismatch(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	// dst too short
	err = plan.Forward(make([]complex64, 4), make([]complex64, 8))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Forward(short dst, src) = %v, want ErrLengthMismatch", err)
	}

	// src too short
	err = plan.Forward(make([]complex64, 8), make([]complex64, 4))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Forward(dst, short src) = %v, want ErrLengthMismatch", err)
	}

	// dst too long
	err = plan.Forward(make([]complex64, 16), make([]complex64, 8))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Forward(long dst, src) = %v, want ErrLengthMismatch", err)
	}
}

func TestInverse_NilSlice(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	err = plan.Inverse(nil, make([]complex64, 8))
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("Inverse(nil, src) = %v, want ErrNilSlice", err)
	}
}

func TestInverse_LengthMismatch(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	err = plan.Inverse(make([]complex64, 4), make([]complex64, 8))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Inverse(short dst, src) = %v, want ErrLengthMismatch", err)
	}
}

func TestInPlace_NilSlice(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	err = plan.InPlace(nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("InPlace(nil) = %v, want ErrNilSlice", err)
	}
}

func TestInPlace_LengthMismatch(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	err = plan.InPlace(make([]complex64, 4))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("InPlace(short) = %v, want ErrLengthMismatch", err)
	}
}

func TestForward_Impulse(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	dst := make([]complex64, 8)
	src := make([]complex64, 8)

	src[0] = 1

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	for i := range dst {
		assertApproxComplex64f(t, dst[i], 1, 1e-4, "dst[%d]", i)
	}
}

func TestForwardInverse_RoundTrip(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	src := make([]complex64, 8)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, 8)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	roundTrip := make([]complex64, 8)

	err = plan.Inverse(roundTrip, dst)
	if err != nil {
		t.Fatalf("Inverse() returned error: %v", err)
	}

	for i := range src {
		assertApproxComplex64f(t, roundTrip[i], src[i], 1e-4, "roundTrip[%d]", i)
	}
}

func TestForwardInverse_Size2(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](2)
	if err != nil {
		t.Fatalf("NewPlan(2) returned error: %v", err)
	}

	src := []complex64{1 + 2i, 3 + 4i}
	dst := make([]complex64, 2)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	roundTrip := make([]complex64, 2)

	err = plan.Inverse(roundTrip, dst)
	if err != nil {
		t.Fatalf("Inverse() returned error: %v", err)
	}

	for i := range src {
		assertApproxComplex64f(t, roundTrip[i], src[i], 1e-4, "roundTrip[%d]", i)
	}
}

func TestForwardInverse_RoundTrip128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex128](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	src := make([]complex128, 8)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, 8)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	roundTrip := make([]complex128, 8)

	err = plan.Inverse(roundTrip, dst)
	if err != nil {
		t.Fatalf("Inverse() returned error: %v", err)
	}

	for i := range src {
		assertApproxComplex128f(t, roundTrip[i], src[i], "roundTrip[%d]", i)
	}
}

func assertApproxComplex64f(t *testing.T, got, want complex64, tol float64, format string, args ...any) {
	t.Helper()

	if cmplx.Abs(complex128(got-want)) > tol {
		t.Fatalf(format+": got %v want %v", append(args, got, want)...)
	}
}

func assertApproxComplex128f(t *testing.T, got, want complex128, format string, args ...any) {
	t.Helper()

	const tol = 1e-10
	if cmplx.Abs(got-want) > tol {
		t.Fatalf(format+": got %v want %v", append(args, got, want)...)
	}
}

// Benchmarks

func BenchmarkNewPlan_64(b *testing.B) {
	for b.Loop() {
		_, _ = NewPlanT[complex64](64)
	}
}

func BenchmarkNewPlan_1024(b *testing.B) {
	for b.Loop() {
		_, _ = NewPlanT[complex64](1024)
	}
}

func BenchmarkNewPlan_65536(b *testing.B) {
	for b.Loop() {
		_, _ = NewPlanT[complex64](65536)
	}
}
