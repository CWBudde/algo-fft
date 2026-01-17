package fft

import (
	"fmt"
	"testing"
)

// TestScaleComplex64InPlace tests scaling of complex64 arrays.
func TestScaleComplex64InPlace(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input []complex64
		scale float32
		want  []complex64
	}{
		{
			name:  "scale by 2",
			input: []complex64{1 + 2i, 3 + 4i, 5 + 6i},
			scale: 2.0,
			want:  []complex64{2 + 4i, 6 + 8i, 10 + 12i},
		},
		{
			name:  "scale by 0.5",
			input: []complex64{2 + 4i, 6 + 8i},
			scale: 0.5,
			want:  []complex64{1 + 2i, 3 + 4i},
		},
		{
			name:  "scale by 1 (identity)",
			input: []complex64{1 + 2i, 3 + 4i, 5 + 6i},
			scale: 1.0,
			want:  []complex64{1 + 2i, 3 + 4i, 5 + 6i},
		},
		{
			name:  "scale by 0 (zero)",
			input: []complex64{1 + 2i, 3 + 4i},
			scale: 0.0,
			want:  []complex64{0, 0},
		},
		{
			name:  "scale by negative",
			input: []complex64{1 + 2i, 3 + 4i},
			scale: -1.0,
			want:  []complex64{-1 - 2i, -3 - 4i},
		},
		{
			name:  "single element",
			input: []complex64{3 + 4i},
			scale: 2.5,
			want:  []complex64{7.5 + 10i},
		},
		{
			name:  "empty array",
			input: []complex64{},
			scale: 2.0,
			want:  []complex64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Make a copy since scaling is in-place
			dst := make([]complex64, len(tt.input))
			copy(dst, tt.input)

			ScaleComplex64InPlace(dst, tt.scale)

			if len(dst) != len(tt.want) {
				t.Fatalf("length mismatch: got %d, want %d", len(dst), len(tt.want))
			}

			for i := range dst {
				if !complexNear64(dst[i], tt.want[i], 1e-5) {
					t.Errorf("dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestScaleComplex128InPlace tests scaling of complex128 arrays.
func TestScaleComplex128InPlace(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input []complex128
		scale float64
		want  []complex128
	}{
		{
			name:  "scale by 2",
			input: []complex128{1 + 2i, 3 + 4i, 5 + 6i},
			scale: 2.0,
			want:  []complex128{2 + 4i, 6 + 8i, 10 + 12i},
		},
		{
			name:  "scale by 0.5",
			input: []complex128{2 + 4i, 6 + 8i},
			scale: 0.5,
			want:  []complex128{1 + 2i, 3 + 4i},
		},
		{
			name:  "scale by 1 (identity)",
			input: []complex128{1 + 2i, 3 + 4i, 5 + 6i},
			scale: 1.0,
			want:  []complex128{1 + 2i, 3 + 4i, 5 + 6i},
		},
		{
			name:  "scale by 0 (zero)",
			input: []complex128{1 + 2i, 3 + 4i},
			scale: 0.0,
			want:  []complex128{0, 0},
		},
		{
			name:  "high precision",
			input: []complex128{1.123456789012345 + 2.234567890123456i},
			scale: 3.345678901234567,
			want:  []complex128{(1.123456789012345 + 2.234567890123456i) * 3.345678901234567},
		},
		{
			name:  "scale by negative",
			input: []complex128{1 + 2i, 3 + 4i},
			scale: -1.0,
			want:  []complex128{-1 - 2i, -3 - 4i},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Make a copy since scaling is in-place
			dst := make([]complex128, len(tt.input))
			copy(dst, tt.input)

			ScaleComplex128InPlace(dst, tt.scale)

			if len(dst) != len(tt.want) {
				t.Fatalf("length mismatch: got %d, want %d", len(dst), len(tt.want))
			}

			for i := range dst {
				if !complexNear128(dst[i], tt.want[i]) {
					t.Errorf("dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestScaleComplex64Large tests scaling with larger arrays (SIMD path).
func TestScaleComplex64Large(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}
	scales := []float32{0.5, 1.0, 2.0, -1.0}

	for _, n := range sizes {
		for _, scale := range scales {
			t.Run(fmt.Sprintf("size_%d_scale_%.1f", n, scale), func(t *testing.T) {
				t.Parallel()

				// Generate test data
				input := make([]complex64, n)
				for i := range input {
					input[i] = complex(float32(i), float32(i+1))
				}

				// Compute expected result
				want := make([]complex64, n)
				for i := range want {
					want[i] = input[i] * complex(scale, 0)
				}

				// Test scaling
				dst := make([]complex64, n)
				copy(dst, input)
				ScaleComplex64InPlace(dst, scale)

				for i := range dst {
					if !complexNear64(dst[i], want[i], 1e-4) {
						t.Errorf("size %d, scale %.1f: dst[%d] = %v, want %v", n, scale, i, dst[i], want[i])
					}
				}
			})
		}
	}
}

// TestScaleComplex128Large tests scaling with larger arrays (SIMD path).
func TestScaleComplex128Large(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}
	scales := []float64{0.5, 1.0, 2.0, -1.0}

	for _, n := range sizes {
		for _, scale := range scales {
			t.Run(fmt.Sprintf("size_%d_scale_%.1f", n, scale), func(t *testing.T) {
				t.Parallel()

				// Generate test data
				input := make([]complex128, n)
				for i := range input {
					input[i] = complex(float64(i), float64(i+1))
				}

				// Compute expected result
				want := make([]complex128, n)
				for i := range want {
					want[i] = input[i] * complex(scale, 0)
				}

				// Test scaling
				dst := make([]complex128, n)
				copy(dst, input)
				ScaleComplex128InPlace(dst, scale)

				for i := range dst {
					if !complexNear128(dst[i], want[i]) {
						t.Errorf("size %d, scale %.1f: dst[%d] = %v, want %v", n, scale, i, dst[i], want[i])
					}
				}
			})
		}
	}
}

// TestScaleIdentityOptimization tests that scale=1.0 is optimized (no-op).
func TestScaleIdentityOptimization(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		input := []complex64{1 + 2i, 3 + 4i, 5 + 6i}
		original := make([]complex64, len(input))
		copy(original, input)

		ScaleComplex64InPlace(input, 1.0)

		// Verify no change
		for i := range input {
			if input[i] != original[i] {
				t.Errorf("identity scaling modified input[%d]: got %v, want %v", i, input[i], original[i])
			}
		}
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		input := []complex128{1 + 2i, 3 + 4i, 5 + 6i}
		original := make([]complex128, len(input))
		copy(original, input)

		ScaleComplex128InPlace(input, 1.0)

		// Verify no change
		for i := range input {
			if input[i] != original[i] {
				t.Errorf("identity scaling modified input[%d]: got %v, want %v", i, input[i], original[i])
			}
		}
	})
}

// Note: complexNear64 and complexNear128 helper functions are defined in complex_mul_test.go
