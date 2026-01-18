package algofft

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

// Test helpers for N-D

func generateRandomNDComplex64(dims []int, seed uint64) []complex64 {
	size := 1
	for _, d := range dims {
		size *= d
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xABCDEF01)) //nolint:gosec

	data := make([]complex64, size)
	for i := range data {
		re := float32(rng.Float64()*20 - 10)
		im := float32(rng.Float64()*20 - 10)
		data[i] = complex(re, im)
	}

	return data
}

func generateRandomNDComplex128(dims []int, seed uint64) []complex128 {
	size := 1
	for _, d := range dims {
		size *= d
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xABCDEF01)) //nolint:gosec

	data := make([]complex128, size)
	for i := range data {
		re := rng.Float64()*20 - 10
		im := rng.Float64()*20 - 10
		data[i] = complex(re, im)
	}

	return data
}

func complexND64NearlyEqual(a, b []complex64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		diff := a[i] - b[i]

		mag := math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff)))
		if mag > tol {
			return false
		}
	}

	return true
}

func complexND128NearlyEqual(a, b []complex128, tol float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		diff := a[i] - b[i]

		mag := math.Sqrt(real(diff)*real(diff) + imag(diff)*imag(diff))
		if mag > tol {
			return false
		}
	}

	return true
}

// Test PlanND creation

func TestNewPlanND(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		dims        []int
		expectError bool
		name        string
	}{
		{[]int{8, 8, 8}, false, "valid_3D"},
		{[]int{4, 8, 16, 32}, false, "valid_4D"},
		{[]int{2, 2, 2, 2, 2}, false, "valid_5D"},
		{[]int{}, true, "empty_dims"},
		{[]int{0, 8, 8}, true, "invalid_zero_dim"},
		{[]int{8, -1, 8}, true, "invalid_negative_dim"},
		{[]int{1}, false, "valid_1D"},
		{[]int{16, 16}, false, "valid_2D"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanND[complex64](testCase.dims)
			if testCase.expectError {
				if err == nil {
					t.Errorf("Expected error for dims %v, got nil", testCase.dims)
				}

				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if plan.NDims() != len(testCase.dims) {
				t.Errorf("NDims: got %d, want %d", plan.NDims(), len(testCase.dims))
			}

			dims := plan.Dims()
			if len(dims) != len(testCase.dims) {
				t.Errorf("Dims length mismatch")
			}

			for i := range dims {
				if dims[i] != testCase.dims[i] {
					t.Errorf("Dims[%d]: got %d, want %d", i, dims[i], testCase.dims[i])
				}
			}

			expectedLen := 1
			for _, d := range testCase.dims {
				expectedLen *= d
			}

			if plan.Len() != expectedLen {
				t.Errorf("Len: got %d, want %d", plan.Len(), expectedLen)
			}
		})
	}
}

// Test round-trip: IFFT(FFT(x)) ≈ x for various dimensions

func TestPlanND_RoundTrip_3D(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		dims []int
		name string
	}{
		{[]int{2, 2, 2}, "2x2x2"},
		{[]int{4, 4, 4}, "4x4x4"},
		{[]int{8, 8, 8}, "8x8x8"},
		{[]int{4, 8, 16}, "4x8x16_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanND32(testCase.dims)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			original := generateRandomNDComplex64(testCase.dims, 12345)
			freq := make([]complex64, len(original))
			roundTrip := make([]complex64, len(original))

			err = plan.Forward(freq, original)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			err = plan.Inverse(roundTrip, freq)
			if err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			tolerance := 1e-3
			if !complexND64NearlyEqual(roundTrip, original, tolerance) {
				t.Errorf("Round-trip failed for dims %v", testCase.dims)
			}
		})
	}
}

func TestPlanND_RoundTrip_4D(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		dims []int
		name string
	}{
		{[]int{2, 2, 2, 2}, "2x2x2x2"},
		{[]int{4, 4, 4, 4}, "4x4x4x4"},
		{[]int{8, 8, 8, 8}, "8x8x8x8"},
		{[]int{2, 4, 8, 16}, "2x4x8x16_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanND32(testCase.dims)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			original := generateRandomNDComplex64(testCase.dims, 54321)
			freq := make([]complex64, len(original))
			roundTrip := make([]complex64, len(original))

			err = plan.Forward(freq, original)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			err = plan.Inverse(roundTrip, freq)
			if err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			tolerance := 1e-3
			if !complexND64NearlyEqual(roundTrip, original, tolerance) {
				t.Errorf("Round-trip failed for dims %v", testCase.dims)
			}
		})
	}
}

func TestPlanND_RoundTrip_5D(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		dims []int
		name string
	}{
		{[]int{2, 2, 2, 2, 2}, "2x2x2x2x2"},
		{[]int{4, 4, 4, 4, 4}, "4x4x4x4x4"},
		{[]int{2, 2, 4, 4, 8}, "2x2x4x4x8_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanND32(testCase.dims)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			original := generateRandomNDComplex64(testCase.dims, 11111)
			freq := make([]complex64, len(original))
			roundTrip := make([]complex64, len(original))

			err = plan.Forward(freq, original)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			err = plan.Inverse(roundTrip, freq)
			if err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			tolerance := 1e-2
			if !complexND64NearlyEqual(roundTrip, original, tolerance) {
				t.Errorf("Round-trip failed for dims %v", testCase.dims)
			}
		})
	}
}

func TestPlanND_RoundTrip_Complex128(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		dims []int
		name string
	}{
		{[]int{4, 4, 4}, "3D_4x4x4"},
		{[]int{2, 2, 2, 2}, "4D_2x2x2x2"},
		{[]int{2, 2, 2, 2, 2}, "5D_2x2x2x2x2"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanND64(testCase.dims)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			original := generateRandomNDComplex128(testCase.dims, 98765)
			freq := make([]complex128, len(original))
			roundTrip := make([]complex128, len(original))

			err = plan.Forward(freq, original)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			err = plan.Inverse(roundTrip, freq)
			if err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			tolerance := 1e-10
			if !complexND128NearlyEqual(roundTrip, original, tolerance) {
				t.Errorf("Round-trip failed for dims %v", testCase.dims)
			}
		})
	}
}

func TestPlanND_BatchStrideRoundTrip(t *testing.T) {
	t.Parallel()

	dims := []int{2, 3, 4}

	size := 1
	for _, d := range dims {
		size *= d
	}

	const (
		batch  = 2
		stride = 32
	)

	plan, err := NewPlanNDWithOptions[complex64](dims, PlanOptions{
		Batch:  batch,
		Stride: stride,
	})
	if err != nil {
		t.Fatalf("NewPlanNDWithOptions failed: %v", err)
	}

	src := make([]complex64, batch*stride)
	dst := make([]complex64, batch*stride)
	roundTrip := make([]complex64, batch*stride)

	for b := range batch {
		signal := generateRandomNDComplex64(dims, uint64(300+b))
		copy(src[b*stride:b*stride+size], signal)
	}

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(roundTrip, dst)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for b := range batch {
		orig := src[b*stride : b*stride+size]

		got := roundTrip[b*stride : b*stride+size]
		if !complexND64NearlyEqual(got, orig, 1e-3) {
			t.Fatalf("batch %d round-trip mismatch", b)
		}
	}
}

// Test that PlanND matches specialized Plan3D for 3D case

func TestPlanND_MatchesPlan3D(t *testing.T) {
	t.Parallel()

	dims := []int{8, 16, 32}

	plan3D, err := NewPlan3D32(dims[0], dims[1], dims[2])
	if err != nil {
		t.Fatalf("Failed to create Plan3D: %v", err)
	}

	planND, err := NewPlanND32(dims)
	if err != nil {
		t.Fatalf("Failed to create PlanND: %v", err)
	}

	signal := generateRandomNDComplex64(dims, 77777)

	// Transform with Plan3D
	out3D := make([]complex64, len(signal))

	err = plan3D.Forward(out3D, signal)
	if err != nil {
		t.Fatalf("Plan3D Forward failed: %v", err)
	}

	// Transform with PlanND
	outND := make([]complex64, len(signal))

	err = planND.Forward(outND, signal)
	if err != nil {
		t.Fatalf("PlanND Forward failed: %v", err)
	}

	// Results should match
	tolerance := 1e-5
	if !complexND64NearlyEqual(out3D, outND, tolerance) {
		t.Errorf("PlanND results differ from Plan3D")

		count := 0
		for i := 0; i < len(out3D) && count < 5; i++ {
			diff := out3D[i] - outND[i]

			mag := math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff)))
			if mag > tolerance {
				t.Errorf("  [%d]: Plan3D=%v, PlanND=%v (diff: %v)", i, out3D[i], outND[i], mag)

				count++
			}
		}
	}
}

// Test in-place operations

func TestPlanND_InPlace(t *testing.T) {
	t.Parallel()

	dims := []int{4, 8, 16}

	plan, err := NewPlanND32(dims)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	data := generateRandomNDComplex64(dims, 55555)
	original := make([]complex64, len(data))
	copy(original, data)

	// Forward in-place
	err = plan.ForwardInPlace(data)
	if err != nil {
		t.Fatalf("ForwardInPlace failed: %v", err)
	}

	// Inverse in-place
	err = plan.InverseInPlace(data)
	if err != nil {
		t.Fatalf("InverseInPlace failed: %v", err)
	}

	// Should recover original
	if !complexND64NearlyEqual(data, original, 1e-3) {
		t.Errorf("In-place round-trip failed")
	}
}

// Test error handling

func TestPlanND_ErrorHandling(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanND32([]int{8, 8, 8})
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	validData := make([]complex64, 8*8*8)
	wrongSize := make([]complex64, 10)

	// Nil slices
	err = plan.Forward(nil, validData)
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("Expected ErrNilSlice for nil dst, got %v", err)
	}

	err = plan.Forward(validData, nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("Expected ErrNilSlice for nil src, got %v", err)
	}

	// Wrong length
	err = plan.Forward(wrongSize, validData)
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Expected ErrLengthMismatch for wrong dst size, got %v", err)
	}

	err = plan.Forward(validData, wrongSize)
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Expected ErrLengthMismatch for wrong src size, got %v", err)
	}
}

// Test Clone

func TestPlanND_Clone(t *testing.T) {
	t.Parallel()

	dims := []int{4, 4, 4}

	original, err := NewPlanND32(dims)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	clone := original.Clone()

	if clone.NDims() != original.NDims() {
		t.Errorf("Clone NDims mismatch")
	}

	// Use both plans concurrently
	data1 := generateRandomNDComplex64(dims, 111)
	data2 := generateRandomNDComplex64(dims, 222)

	out1 := make([]complex64, len(data1))
	out2 := make([]complex64, len(data2))

	done := make(chan bool, 2)

	go func() {
		_ = original.Forward(out1, data1)

		done <- true
	}()

	go func() {
		_ = clone.Forward(out2, data2)

		done <- true
	}()

	<-done
	<-done

	// Results should be different (different inputs)
	if complexND64NearlyEqual(out1, out2, 1e-10) {
		t.Errorf("Clone produced identical results for different inputs")
	}
}

// Test String representation

func TestPlanND_String(t *testing.T) {
	t.Parallel()

	plan3D, _ := NewPlanND32([]int{8, 16, 32})

	str3D := plan3D.String()
	if str3D != "PlanND[complex64](8x16x32)" {
		t.Errorf("String mismatch for 3D: got %q", str3D)
	}

	plan4D, _ := NewPlanND32([]int{2, 4, 8, 16})

	str4D := plan4D.String()
	if str4D != "PlanND[complex64](2x4x8x16)" {
		t.Errorf("String mismatch for 4D: got %q", str4D)
	}

	plan64, _ := NewPlanND64([]int{4, 8})

	str64 := plan64.String()
	if str64 != "PlanND[complex128](4x8)" {
		t.Errorf("String mismatch for complex128: got %q", str64)
	}
}

// Benchmarks

func BenchmarkPlanND_3D_8x8x8(b *testing.B) {
	dims := []int{8, 8, 8}
	plan, _ := NewPlanND32(dims)
	signal := generateRandomNDComplex64(dims, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

func BenchmarkPlanND_4D_4x4x4x4(b *testing.B) {
	dims := []int{4, 4, 4, 4}
	plan, _ := NewPlanND32(dims)
	signal := generateRandomNDComplex64(dims, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

func BenchmarkPlanND_5D_2x2x2x2x2(b *testing.B) {
	dims := []int{2, 2, 2, 2, 2}
	plan, _ := NewPlanND32(dims)
	signal := generateRandomNDComplex64(dims, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

// Benchmark PlanND vs Plan3D overhead

func BenchmarkPlan3D_8x8x8(b *testing.B) {
	plan, _ := NewPlan3D32(8, 8, 8)
	signal := generateRandom3DComplex64(8, 8, 8, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

func BenchmarkPlanND_vs_Plan3D_8x8x8(b *testing.B) {
	plan, _ := NewPlanND32([]int{8, 8, 8})
	signal := generateRandomNDComplex64([]int{8, 8, 8}, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}
