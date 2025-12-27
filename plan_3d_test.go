package algoforge

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/MeKo-Christian/algoforge/internal/reference"
)

// Test helpers for 3D

func generateRandom3DComplex64(depth, height, width int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xCAFEBABE)) //nolint:gosec // Intentionally non-crypto for reproducible tests

	data := make([]complex64, depth*height*width)
	for i := range data {
		re := float32(rng.Float64()*20 - 10)
		im := float32(rng.Float64()*20 - 10)
		data[i] = complex(re, im)
	}

	return data
}

func generateRandom3DComplex128(depth, height, width int, seed uint64) []complex128 {
	rng := rand.New(rand.NewPCG(seed, seed^0xCAFEBABE)) //nolint:gosec // Intentionally non-crypto for reproducible tests

	data := make([]complex128, depth*height*width)
	for i := range data {
		re := rng.Float64()*20 - 10
		im := rng.Float64()*20 - 10
		data[i] = complex(re, im)
	}

	return data
}

func complex3D64NearlyEqual(a, b []complex64, tol float64) bool {
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

func complex3D128NearlyEqual(a, b []complex128, tol float64) bool {
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

// Test Plan3D creation

func TestNewPlan3D(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		expectError          bool
		name                 string
	}{
		{8, 8, 8, false, "valid_8x8x8"},
		{4, 8, 16, false, "valid_4x8x16"},
		{0, 8, 8, true, "invalid_zero_depth"},
		{8, 0, 8, true, "invalid_zero_height"},
		{8, 8, 0, true, "invalid_zero_width"},
		{-1, 8, 8, true, "invalid_negative_depth"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan3D[complex64](tc.depth, tc.height, tc.width)
			if tc.expectError {
				if err == nil {
					t.Errorf("Expected error for dimensions %dx%dx%d, got nil", tc.depth, tc.height, tc.width)
				}

				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if plan.Depth() != tc.depth {
				t.Errorf("Depth: got %d, want %d", plan.Depth(), tc.depth)
			}

			if plan.Height() != tc.height {
				t.Errorf("Height: got %d, want %d", plan.Height(), tc.height)
			}

			if plan.Width() != tc.width {
				t.Errorf("Width: got %d, want %d", plan.Width(), tc.width)
			}

			if plan.Len() != tc.depth*tc.height*tc.width {
				t.Errorf("Len: got %d, want %d", plan.Len(), tc.depth*tc.height*tc.width)
			}
		})
	}
}

// Test round-trip: IFFT(FFT(x)) ≈ x

func TestPlan3D_RoundTrip_Complex64(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		name                 string
	}{
		{2, 2, 2, "2x2x2"},
		{4, 4, 4, "4x4x4"},
		{8, 8, 8, "8x8x8"},
		{16, 16, 16, "16x16x16"},
		{4, 8, 16, "4x8x16_nonsquare"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan3D32(tc.depth, tc.height, tc.width)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			original := generateRandom3DComplex64(tc.depth, tc.height, tc.width, 98765)
			freq := make([]complex64, len(original))
			roundTrip := make([]complex64, len(original))

			// Forward
			if err := plan.Forward(freq, original); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Inverse
			if err := plan.Inverse(roundTrip, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Verify round-trip
			tolerance := 1e-3
			if !complex3D64NearlyEqual(roundTrip, original, tolerance) {
				t.Errorf("Round-trip failed for %dx%dx%d", tc.depth, tc.height, tc.width)
				// Show first few mismatches
				count := 0
				for i := 0; i < len(original) && count < 5; i++ {
					diff := roundTrip[i] - original[i]

					mag := math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff)))
					if mag > tolerance {
						t.Errorf("  [%d]: got %v, want %v (diff: %v)", i, roundTrip[i], original[i], mag)

						count++
					}
				}
			}
		})
	}
}

func TestPlan3D_RoundTrip_Complex128(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		name                 string
	}{
		{2, 2, 2, "2x2x2"},
		{4, 4, 4, "4x4x4"},
		{8, 8, 8, "8x8x8"},
		{4, 8, 16, "4x8x16_nonsquare"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan3D64(tc.depth, tc.height, tc.width)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			original := generateRandom3DComplex128(tc.depth, tc.height, tc.width, 98765)
			freq := make([]complex128, len(original))
			roundTrip := make([]complex128, len(original))

			if err := plan.Forward(freq, original); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if err := plan.Inverse(roundTrip, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			tolerance := 1e-10
			if !complex3D128NearlyEqual(roundTrip, original, tolerance) {
				t.Errorf("Round-trip failed for %dx%dx%d", tc.depth, tc.height, tc.width)
			}
		})
	}
}

// Test against reference naive DFT

func TestPlan3D_VsReference_Complex64(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		name                 string
	}{
		{2, 2, 2, "2x2x2"},
		{4, 4, 4, "4x4x4"},
		{8, 8, 8, "8x8x8"},
		{4, 8, 16, "4x8x16_nonsquare"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan3D32(tc.depth, tc.height, tc.width)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			signal := generateRandom3DComplex64(tc.depth, tc.height, tc.width, 11111)

			// Optimized FFT
			fast := make([]complex64, len(signal))
			if err := plan.Forward(fast, signal); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Reference naive DFT
			naive := reference.NaiveDFT3D(signal, tc.depth, tc.height, tc.width)

			// Compare
			tolerance := 1e-2
			if !complex3D64NearlyEqual(fast, naive, tolerance) {
				t.Errorf("FFT result differs from reference for %dx%dx%d", tc.depth, tc.height, tc.width)

				count := 0
				for i := 0; i < len(fast) && count < 5; i++ {
					diff := fast[i] - naive[i]

					mag := math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff)))
					if mag > tolerance {
						t.Errorf("  [%d]: fast=%v, naive=%v (diff: %v)", i, fast[i], naive[i], mag)

						count++
					}
				}
			}
		})
	}
}

func TestPlan3D_BatchStrideForward(t *testing.T) {
	t.Parallel()

	const (
		depth  = 2
		height = 3
		width  = 4
		batch  = 2
		stride = depth*height*width + 7
	)

	plan, err := NewPlan3DWithOptions[complex64](depth, height, width, PlanOptions{
		Batch:  batch,
		Stride: stride,
	})
	if err != nil {
		t.Fatalf("NewPlan3DWithOptions failed: %v", err)
	}

	src := make([]complex64, batch*stride)
	dst := make([]complex64, batch*stride)

	signals := make([][]complex64, batch)
	for b := 0; b < batch; b++ {
		signal := generateRandom3DComplex64(depth, height, width, uint64(200+b))
		signals[b] = signal
		copy(src[b*stride:b*stride+depth*height*width], signal)
	}

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	tol := 1e-3
	for b := 0; b < batch; b++ {
		want := reference.NaiveDFT3D(signals[b], depth, height, width)
		got := dst[b*stride : b*stride+depth*height*width]
		if !complex3D64NearlyEqual(got, want, tol) {
			t.Fatalf("batch %d result differs from reference", b)
		}
	}
}

func TestPlan3D_VsReference_Complex128(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		name                 string
	}{
		{2, 2, 2, "2x2x2"},
		{4, 4, 4, "4x4x4"},
		{8, 8, 8, "8x8x8"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan3D64(tc.depth, tc.height, tc.width)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			signal := generateRandom3DComplex128(tc.depth, tc.height, tc.width, 11111)

			fast := make([]complex128, len(signal))
			if err := plan.Forward(fast, signal); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			naive := reference.NaiveDFT3D128(signal, tc.depth, tc.height, tc.width)

			tolerance := 1e-9
			if !complex3D128NearlyEqual(fast, naive, tolerance) {
				t.Errorf("FFT result differs from reference for %dx%dx%d", tc.depth, tc.height, tc.width)
			}
		})
	}
}

// Test in-place operations

func TestPlan3D_InPlace(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan3D32(8, 8, 8)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	data := generateRandom3DComplex64(8, 8, 8, 55555)
	original := make([]complex64, len(data))
	copy(original, data)

	// Forward in-place
	if err := plan.ForwardInPlace(data); err != nil {
		t.Fatalf("ForwardInPlace failed: %v", err)
	}

	// Inverse in-place
	if err := plan.InverseInPlace(data); err != nil {
		t.Fatalf("InverseInPlace failed: %v", err)
	}

	// Should recover original
	if !complex3D64NearlyEqual(data, original, 1e-3) {
		t.Errorf("In-place round-trip failed")
	}
}

// Test error handling

func TestPlan3D_ErrorHandling(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan3D32(8, 8, 8)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	validData := make([]complex64, 8*8*8)
	wrongSize := make([]complex64, 10)

	// Nil slices
	if err := plan.Forward(nil, validData); !errors.Is(err, ErrNilSlice) {
		t.Errorf("Expected ErrNilSlice for nil dst, got %v", err)
	}

	if err := plan.Forward(validData, nil); !errors.Is(err, ErrNilSlice) {
		t.Errorf("Expected ErrNilSlice for nil src, got %v", err)
	}

	// Wrong length
	if err := plan.Forward(wrongSize, validData); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Expected ErrLengthMismatch for wrong dst size, got %v", err)
	}

	if err := plan.Forward(validData, wrongSize); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("Expected ErrLengthMismatch for wrong src size, got %v", err)
	}
}

// Test Clone

func TestPlan3D_Clone(t *testing.T) {
	t.Parallel()

	original, err := NewPlan3D32(8, 8, 8)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	clone := original.Clone()

	if clone.Depth() != original.Depth() {
		t.Errorf("Clone depth mismatch")
	}

	if clone.Height() != original.Height() {
		t.Errorf("Clone height mismatch")
	}

	if clone.Width() != original.Width() {
		t.Errorf("Clone width mismatch")
	}

	// Use both plans concurrently
	data1 := generateRandom3DComplex64(8, 8, 8, 111)
	data2 := generateRandom3DComplex64(8, 8, 8, 222)

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
	if complex3D64NearlyEqual(out1, out2, 1e-10) {
		t.Errorf("Clone produced identical results for different inputs")
	}
}

// Test String representation

func TestPlan3D_String(t *testing.T) {
	t.Parallel()

	plan32, _ := NewPlan3D32(8, 16, 32)

	str32 := plan32.String()
	if str32 != "Plan3D[complex64](8x16x32)" {
		t.Errorf("String mismatch for complex64: got %q", str32)
	}

	plan64, _ := NewPlan3D64(4, 8, 16)

	str64 := plan64.String()
	if str64 != "Plan3D[complex128](4x8x16)" {
		t.Errorf("String mismatch for complex128: got %q", str64)
	}
}

// Benchmarks

func BenchmarkPlan3D_Forward_8x8x8(b *testing.B) {
	plan, _ := NewPlan3D32(8, 8, 8)
	signal := generateRandom3DComplex64(8, 8, 8, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8)) // 8 bytes per complex64

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

func BenchmarkPlan3D_Forward_16x16x16(b *testing.B) {
	plan, _ := NewPlan3D32(16, 16, 16)
	signal := generateRandom3DComplex64(16, 16, 16, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

func BenchmarkPlan3D_Forward_32x32x32(b *testing.B) {
	plan, _ := NewPlan3D32(32, 32, 32)
	signal := generateRandom3DComplex64(32, 32, 32, 111)
	freq := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(signal) * 8))

	for range b.N {
		_ = plan.Forward(freq, signal)
	}
}

func BenchmarkPlan3D_RoundTrip_16x16x16(b *testing.B) {
	plan, _ := NewPlan3D32(16, 16, 16)
	signal := generateRandom3DComplex64(16, 16, 16, 111)
	freq := make([]complex64, len(signal))
	recovered := make([]complex64, len(signal))

	b.ResetTimer()
	b.ReportAllocs()

	for range b.N {
		_ = plan.Forward(freq, signal)
		_ = plan.Inverse(recovered, freq)
	}
}
