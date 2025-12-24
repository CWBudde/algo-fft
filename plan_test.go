package algoforge

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
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Errorf("NewPlan(%d) returned error: %v", n, err)
			continue
		}

		if plan.Len() != n {
			t.Errorf("NewPlan(%d).Len() = %d, want %d", n, plan.Len(), n)
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

	invalidSizes := []int{0, -1, 3, 5, 6, 7, 9, 10, 12, 15, 17, 100, 1000}
	for _, n := range invalidSizes {
		plan, err := NewPlan[complex64](n)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("NewPlan(%d) = (%v, %v), want (nil, ErrInvalidLength)", n, plan, err)
		}
	}
}

func TestNewPlan_TwiddleFactors(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex128](8)
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

	plan, err := NewPlan[complex128](16)
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

func TestNewPlan_BitReversalIndices(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	// For n=8 (3 bits), expected bit-reversal permutation:
	// 0 (000) -> 0 (000)
	// 1 (001) -> 4 (100)
	// 2 (010) -> 2 (010)
	// 3 (011) -> 6 (110)
	// 4 (100) -> 1 (001)
	// 5 (101) -> 5 (101)
	// 6 (110) -> 3 (011)
	// 7 (111) -> 7 (111)
	expected := []int{0, 4, 2, 6, 1, 5, 3, 7}

	for i, exp := range expected {
		if plan.bitrev[i] != exp {
			t.Errorf("bitrev[%d] = %d, want %d", i, plan.bitrev[i], exp)
		}
	}
}

func TestNewPlan_BitReversalInvolution(t *testing.T) {
	t.Parallel()

	// Bit reversal applied twice should give identity
	plan, err := NewPlan[complex64](64)
	if err != nil {
		t.Fatalf("NewPlan(64) returned error: %v", err)
	}

	for i := range plan.bitrev {
		j := plan.bitrev[i]
		k := plan.bitrev[j]

		if k != i {
			t.Errorf("bitrev[bitrev[%d]] = %d, want %d (involution property)", i, k, i)
		}
	}
}

// Tests for Forward/Inverse methods and error handling

func TestForward_NilSlice(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](8)
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

	plan, err := NewPlan[complex64](8)
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

	plan, err := NewPlan[complex64](8)
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

	plan, err := NewPlan[complex64](8)
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

	plan, err := NewPlan[complex64](8)
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

	plan, err := NewPlan[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	err = plan.InPlace(make([]complex64, 4))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("InPlace(short) = %v, want ErrLengthMismatch", err)
	}
}

func TestForward_ReturnsNotImplemented(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	dst := make([]complex64, 8)
	src := make([]complex64, 8)

	err = plan.Forward(dst, src)
	if !errors.Is(err, ErrNotImplemented) {
		t.Errorf("Forward() = %v, want ErrNotImplemented", err)
	}
}

func TestInverse_ReturnsNotImplemented(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan(8) returned error: %v", err)
	}

	dst := make([]complex64, 8)
	src := make([]complex64, 8)

	err = plan.Inverse(dst, src)
	if !errors.Is(err, ErrNotImplemented) {
		t.Errorf("Inverse() = %v, want ErrNotImplemented", err)
	}
}

func TestForwardInverse_Size2_Asm(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](2)
	if err != nil {
		t.Fatalf("NewPlan(2) returned error: %v", err)
	}

	src := []complex64{1 + 2i, 3 + 4i}
	dst := make([]complex64, 2)

	err = plan.Forward(dst, src)
	if err == nil {
		roundTrip := make([]complex64, 2)
		err = plan.Inverse(roundTrip, dst)
		if err != nil {
			t.Fatalf("Inverse() returned error: %v", err)
		}

		for i := range src {
			if roundTrip[i] != src[i] {
				t.Fatalf("roundTrip[%d] = %v, want %v", i, roundTrip[i], src[i])
			}
		}
		return
	}

	if !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Forward() = %v, want ErrNotImplemented or nil (asm)", err)
	}
}

// Benchmarks

func BenchmarkNewPlan_64(b *testing.B) {
	for b.Loop() {
		_, _ = NewPlan[complex64](64)
	}
}

func BenchmarkNewPlan_1024(b *testing.B) {
	for b.Loop() {
		_, _ = NewPlan[complex64](1024)
	}
}

func BenchmarkNewPlan_65536(b *testing.B) {
	for b.Loop() {
		_, _ = NewPlan[complex64](65536)
	}
}
