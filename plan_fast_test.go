package algofft

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// TestFastPlan_PowersOfTwo verifies FastPlan works for all power-of-2 sizes
// that have codelets registered.
func TestFastPlan_PowersOfTwo(t *testing.T) {
	t.Parallel()

	// Sizes that should have codelets
	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
	for _, n := range sizes {
		plan, err := NewFastPlan[complex64](n)
		if errors.Is(err, ErrNotImplemented) {
			t.Logf("Skipping size %d: no codelet available", n)
			continue
		}

		if err != nil {
			t.Errorf("NewFastPlan(%d) returned error: %v", n, err)
			continue
		}

		if plan.Len() != n {
			t.Errorf("NewFastPlan(%d).Len() = %d, want %d", n, plan.Len(), n)
		}

		// Verify Forward works
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i), 0)
		}

		dst := make([]complex64, n)
		plan.Forward(dst, src)

		// Verify non-trivial output (at least DC bin should be non-zero)
		if dst[0] == 0 {
			t.Errorf("FastPlan(%d).Forward produced zero DC bin", n)
		}
	}
}

// TestFastPlan_InvalidSizes verifies FastPlan rejects invalid sizes.
func TestFastPlan_InvalidSizes(t *testing.T) {
	t.Parallel()

	invalidSizes := []int{0, -1, 3, 5, 7, 9, 11} // Non-power-of-2 and negative
	for _, n := range invalidSizes {
		plan, err := NewFastPlan[complex64](n)
		if plan != nil {
			t.Errorf("NewFastPlan(%d) should return nil plan", n)
		}

		if err == nil {
			t.Errorf("NewFastPlan(%d) should return error", n)
		}
	}
}

// TestFastPlan_MatchesSafeAPI verifies FastPlan produces identical results to Plan.
func TestFastPlan_MatchesSafeAPI(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 64, 256}
	for _, n := range sizes {
		fastPlan, err := NewFastPlan[complex64](n)
		if errors.Is(err, ErrNotImplemented) {
			t.Logf("Skipping size %d: no codelet available", n)
			continue
		}

		if err != nil {
			t.Fatalf("NewFastPlan(%d) error: %v", n, err)
		}

		safePlan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlanT(%d) error: %v", n, err)
		}

		// Create test input
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i%7)-3, float32(i%5)-2)
		}

		fastDst := make([]complex64, n)
		safeDst := make([]complex64, n)

		fastPlan.Forward(fastDst, src)

		if err := safePlan.Forward(safeDst, src); err != nil {
			t.Fatalf("Safe plan forward error: %v", err)
		}

		// Compare outputs
		for i := range n {
			diff := cmplx.Abs(complex128(fastDst[i] - safeDst[i]))
			if diff > 1e-5 {
				t.Errorf("Size %d: Forward mismatch at bin %d: fast=%v safe=%v diff=%v",
					n, i, fastDst[i], safeDst[i], diff)
			}
		}
	}
}

// TestFastPlan_RoundTrip verifies Inverse(Forward(x)) ≈ x.
func TestFastPlan_RoundTrip(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 64, 256}
	for _, n := range sizes {
		plan, err := NewFastPlan[complex64](n)
		if errors.Is(err, ErrNotImplemented) {
			continue
		}

		if err != nil {
			t.Fatalf("NewFastPlan(%d) error: %v", n, err)
		}

		// Create test input
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i%7)-3, float32(i%5)-2)
		}

		freq := make([]complex64, n)
		result := make([]complex64, n)

		plan.Forward(freq, src)
		plan.Inverse(result, freq)

		// Compare with original
		for i := range n {
			diff := cmplx.Abs(complex128(result[i] - src[i]))
			if diff > 1e-4 {
				t.Errorf("Size %d: RoundTrip mismatch at %d: got=%v want=%v diff=%v",
					n, i, result[i], src[i], diff)
			}
		}
	}
}

// TestFastPlan_InPlace verifies in-place transforms work correctly.
func TestFastPlan_InPlace(t *testing.T) {
	t.Parallel()

	n := 64

	plan, err := NewFastPlan[complex64](n)
	if errors.Is(err, ErrNotImplemented) {
		t.Skip("No codelet for size 64")
	}

	if err != nil {
		t.Fatalf("NewFastPlan(%d) error: %v", n, err)
	}

	// Create test input
	data := make([]complex64, n)
	for i := range data {
		data[i] = complex(float32(i), 0)
	}

	original := make([]complex64, n)
	copy(original, data)

	plan.InPlace(data)
	plan.InverseInPlace(data)

	// Compare with original
	for i := range n {
		diff := cmplx.Abs(complex128(data[i] - original[i]))
		if diff > 1e-4 {
			t.Errorf("InPlace roundtrip mismatch at %d: got=%v want=%v", i, data[i], original[i])
		}
	}
}

// TestFastPlanReal32_Creation verifies FastPlanReal32 creation.
func TestFastPlanReal32_Creation(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512, 1024}
	for _, n := range sizes {
		plan, err := NewFastPlanReal32(n)
		if errors.Is(err, ErrNotImplemented) {
			t.Logf("Skipping size %d: no codelet available", n)
			continue
		}

		if err != nil {
			t.Errorf("NewFastPlanReal32(%d) error: %v", n, err)
			continue
		}

		if plan.Len() != n {
			t.Errorf("NewFastPlanReal32(%d).Len() = %d, want %d", n, plan.Len(), n)
		}

		if plan.SpectrumLen() != n/2+1 {
			t.Errorf("NewFastPlanReal32(%d).SpectrumLen() = %d, want %d", n, plan.SpectrumLen(), n/2+1)
		}
	}
}

// TestFastPlanReal32_InvalidSizes verifies FastPlanReal32 rejects invalid sizes.
func TestFastPlanReal32_InvalidSizes(t *testing.T) {
	t.Parallel()

	invalidSizes := []int{0, 1, 3, 5, 7} // Too small or odd
	for _, n := range invalidSizes {
		plan, err := NewFastPlanReal32(n)
		if plan != nil {
			t.Errorf("NewFastPlanReal32(%d) should return nil plan", n)
		}

		if err == nil {
			t.Errorf("NewFastPlanReal32(%d) should return error", n)
		}
	}
}

// TestFastPlanReal32_MatchesSafeAPI verifies FastPlanReal32 matches PlanRealT.
func TestFastPlanReal32_MatchesSafeAPI(t *testing.T) {
	t.Parallel()

	sizes := []int{32, 64, 256}
	for _, n := range sizes {
		fastPlan, err := NewFastPlanReal32(n)
		if errors.Is(err, ErrNotImplemented) {
			t.Logf("Skipping size %d: no codelet available", n)
			continue
		}

		if err != nil {
			t.Fatalf("NewFastPlanReal32(%d) error: %v", n, err)
		}

		safePlan, err := NewPlanRealT[float32, complex64](n)
		if err != nil {
			t.Fatalf("NewPlanRealT(%d) error: %v", n, err)
		}

		// Create test input
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i%7) - 3
		}

		fastDst := make([]complex64, n/2+1)
		safeDst := make([]complex64, n/2+1)

		fastPlan.Forward(fastDst, src)

		if err := safePlan.Forward(safeDst, src); err != nil {
			t.Fatalf("Safe plan forward error: %v", err)
		}

		// Compare outputs
		for i := range fastDst {
			diff := cmplx.Abs(complex128(fastDst[i] - safeDst[i]))
			if diff > 1e-4 {
				t.Errorf("Size %d: Forward mismatch at bin %d: fast=%v safe=%v diff=%v",
					n, i, fastDst[i], safeDst[i], diff)
			}
		}
	}
}

// TestFastPlanReal32_RoundTrip verifies Inverse(Forward(x)) ≈ x.
func TestFastPlanReal32_RoundTrip(t *testing.T) {
	t.Parallel()

	sizes := []int{32, 64, 256}
	for _, n := range sizes {
		plan, err := NewFastPlanReal32(n)
		if errors.Is(err, ErrNotImplemented) {
			continue
		}

		if err != nil {
			t.Fatalf("NewFastPlanReal32(%d) error: %v", n, err)
		}

		// Create test input
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i%7) - 3
		}

		freq := make([]complex64, n/2+1)
		result := make([]float32, n)

		plan.Forward(freq, src)
		plan.Inverse(result, freq)

		// Compare with original
		for i := range n {
			diff := math.Abs(float64(result[i] - src[i]))
			if diff > 1e-4 {
				t.Errorf("Size %d: RoundTrip mismatch at %d: got=%v want=%v diff=%v",
					n, i, result[i], src[i], diff)
			}
		}
	}
}

// TestFastPlanReal64_RoundTrip verifies FastPlanReal64 Inverse(Forward(x)) ≈ x.
func TestFastPlanReal64_RoundTrip(t *testing.T) {
	t.Parallel()

	sizes := []int{32, 64, 256}
	for _, n := range sizes {
		plan, err := NewFastPlanReal64(n)
		if errors.Is(err, ErrNotImplemented) {
			continue
		}

		if err != nil {
			t.Fatalf("NewFastPlanReal64(%d) error: %v", n, err)
		}

		// Create test input
		src := make([]float64, n)
		for i := range src {
			src[i] = float64(i%7) - 3
		}

		freq := make([]complex128, n/2+1)
		result := make([]float64, n)

		plan.Forward(freq, src)
		plan.Inverse(result, freq)

		// Compare with original (tighter tolerance for float64)
		for i := range n {
			diff := math.Abs(result[i] - src[i])
			if diff > 1e-10 {
				t.Errorf("Size %d: RoundTrip mismatch at %d: got=%v want=%v diff=%v",
					n, i, result[i], src[i], diff)
			}
		}
	}
}

// BenchmarkFastPlan_vs_Plan compares FastPlan to regular Plan performance.
func BenchmarkFastPlan_vs_Plan(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, n := range sizes {
		fastPlan, err := NewFastPlan[complex64](n)
		if err != nil {
			b.Logf("Skipping FastPlan size %d: %v", n, err)
			continue
		}

		safePlan, err := NewPlanT[complex64](n)
		if err != nil {
			b.Fatalf("NewPlanT(%d) error: %v", n, err)
		}

		src := make([]complex64, n)
		dst := make([]complex64, n)

		b.Run("Fast/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				fastPlan.Forward(dst, src)
			}
		})

		b.Run("Safe/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				_ = safePlan.Forward(dst, src)
			}
		})
	}
}

// BenchmarkFastPlanReal32_vs_PlanRealT compares real FFT performance.
func BenchmarkFastPlanReal32_vs_PlanRealT(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, n := range sizes {
		fastPlan, err := NewFastPlanReal32(n)
		if err != nil {
			b.Logf("Skipping FastPlanReal32 size %d: %v", n, err)
			continue
		}

		safePlan, err := NewPlanRealT[float32, complex64](n)
		if err != nil {
			b.Fatalf("NewPlanRealT(%d) error: %v", n, err)
		}

		src := make([]float32, n)
		dst := make([]complex64, n/2+1)

		b.Run("Fast/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				fastPlan.Forward(dst, src)
			}
		})

		b.Run("Safe/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				_ = safePlan.Forward(dst, src)
			}
		})
	}
}
