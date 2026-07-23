package algofft

import (
	"math/cmplx"
	"testing"
)

// TestForwardInPlace_MatchesForward verifies the 1D ForwardInPlace/
// InverseInPlace pair matches Forward/Inverse.
func TestForwardInPlace_MatchesForward(t *testing.T) {
	t.Parallel()

	const n = 64

	plan, err := NewPlan[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i%5)-2, float32(i%3)-1)
	}

	want := make([]complex64, n)
	if err := plan.Forward(want, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	got := make([]complex64, n)
	copy(got, src)

	if err := plan.ForwardInPlace(got); err != nil {
		t.Fatalf("ForwardInPlace failed: %v", err)
	}

	for i := range want {
		if cmplx.Abs(complex128(got[i]-want[i])) > 1e-5 {
			t.Fatalf("ForwardInPlace mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}

	if err := plan.InverseInPlace(got); err != nil {
		t.Fatalf("InverseInPlace failed: %v", err)
	}

	for i := range src {
		if cmplx.Abs(complex128(got[i]-src[i])) > 1e-5 {
			t.Fatalf("round-trip mismatch at %d: got %v, want %v", i, got[i], src[i])
		}
	}
}

// TestFastPlanForwardInPlace verifies that FastPlan's ForwardInPlace
// produces the Forward result.
func TestFastPlanForwardInPlace(t *testing.T) {
	t.Parallel()

	const n = 64

	fp, err := NewFastPlan[complex64](n)
	if err != nil {
		t.Skipf("no codelet for size %d on this build: %v", n, err)
	}

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i%7)-3, float32(i%4)-2)
	}

	want := make([]complex64, n)
	fp.Forward(want, src)

	got := make([]complex64, n)
	copy(got, src)
	fp.ForwardInPlace(got)

	for i := range want {
		if cmplx.Abs(complex128(got[i]-want[i])) > 1e-5 {
			t.Fatalf("ForwardInPlace mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
