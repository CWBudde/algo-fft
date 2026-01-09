package fft

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// TestComplexMulArrayComplex64 tests element-wise complex multiplication for complex64.
func TestComplexMulArrayComplex64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		a    []complex64
		b    []complex64
		want []complex64
	}{
		{
			name: "simple multiplication",
			a:    []complex64{1 + 2i, 3 + 4i},
			b:    []complex64{5 + 6i, 7 + 8i},
			want: []complex64{(1 + 2i) * (5 + 6i), (3 + 4i) * (7 + 8i)}, // -7+16i, -11+52i
		},
		{
			name: "identity multiplication",
			a:    []complex64{1 + 2i, 3 + 4i, 5 + 6i},
			b:    []complex64{1, 1, 1},
			want: []complex64{1 + 2i, 3 + 4i, 5 + 6i},
		},
		{
			name: "zero multiplication",
			a:    []complex64{1 + 2i, 3 + 4i},
			b:    []complex64{0, 0},
			want: []complex64{0, 0},
		},
		{
			name: "single element",
			a:    []complex64{2 + 3i},
			b:    []complex64{4 + 5i},
			want: []complex64{(2 + 3i) * (4 + 5i)}, // -7+22i
		},
		{
			name: "power of 2 length (8)",
			a:    []complex64{1, 2, 3, 4, 5, 6, 7, 8},
			b:    []complex64{1i, 2i, 3i, 4i, 5i, 6i, 7i, 8i},
			want: []complex64{1i, 4i, 9i, 16i, 25i, 36i, 49i, 64i},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			dst := make([]complex64, len(testCase.a))
			ComplexMulArrayComplex64(dst, testCase.a, testCase.b)

			for i := range dst {
				if !complexNear64(dst[i], testCase.want[i], 1e-5) {
					t.Errorf("dst[%d] = %v, want %v", i, dst[i], testCase.want[i])
				}
			}
		})
	}
}

// TestComplexMulArrayComplex128 tests element-wise complex multiplication for complex128.
func TestComplexMulArrayComplex128(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		a    []complex128
		b    []complex128
		want []complex128
	}{
		{
			name: "simple multiplication",
			a:    []complex128{1 + 2i, 3 + 4i},
			b:    []complex128{5 + 6i, 7 + 8i},
			want: []complex128{(1 + 2i) * (5 + 6i), (3 + 4i) * (7 + 8i)},
		},
		{
			name: "identity multiplication",
			a:    []complex128{1 + 2i, 3 + 4i, 5 + 6i},
			b:    []complex128{1, 1, 1},
			want: []complex128{1 + 2i, 3 + 4i, 5 + 6i},
		},
		{
			name: "high precision",
			a:    []complex128{1.123456789012345 + 2.234567890123456i},
			b:    []complex128{3.345678901234567 + 4.456789012345678i},
			want: []complex128{(1.123456789012345 + 2.234567890123456i) * (3.345678901234567 + 4.456789012345678i)},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			dst := make([]complex128, len(testCase.a))
			ComplexMulArrayComplex128(dst, testCase.a, testCase.b)

			for i := range dst {
				if !complexNear128(dst[i], testCase.want[i], 1e-10) {
					t.Errorf("dst[%d] = %v, want %v", i, dst[i], testCase.want[i])
				}
			}
		})
	}
}

// TestComplexMulArrayInPlaceComplex64 tests in-place element-wise multiplication.
func TestComplexMulArrayInPlaceComplex64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		dst  []complex64
		src  []complex64
		want []complex64
	}{
		{
			name: "simple in-place",
			dst:  []complex64{1 + 2i, 3 + 4i},
			src:  []complex64{5 + 6i, 7 + 8i},
			want: []complex64{(1 + 2i) * (5 + 6i), (3 + 4i) * (7 + 8i)},
		},
		{
			name: "convolution-like length",
			dst:  []complex64{1, 2, 3, 4, 5, 6, 7},
			src:  []complex64{1, 1, 1, 1, 1, 1, 1},
			want: []complex64{1, 2, 3, 4, 5, 6, 7},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			// Copy dst since it will be modified in-place
			dst := make([]complex64, len(testCase.dst))
			copy(dst, testCase.dst)

			ComplexMulArrayInPlaceComplex64(dst, testCase.src)

			for i := range dst {
				if !complexNear64(dst[i], testCase.want[i], 1e-5) {
					t.Errorf("dst[%d] = %v, want %v", i, dst[i], testCase.want[i])
				}
			}
		})
	}
}

// TestComplexMulArrayInPlaceComplex128 tests in-place element-wise multiplication for complex128.
func TestComplexMulArrayInPlaceComplex128(t *testing.T) {
	t.Parallel()

	dst := []complex128{1 + 2i, 3 + 4i}
	src := []complex128{5 + 6i, 7 + 8i}
	want := []complex128{(1 + 2i) * (5 + 6i), (3 + 4i) * (7 + 8i)}

	ComplexMulArrayInPlaceComplex128(dst, src)

	for i := range dst {
		if !complexNear128(dst[i], want[i], 1e-10) {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// TestComplexMulArrayLargeComplex64 tests larger arrays that would benefit from SIMD.
func TestComplexMulArrayLargeComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			t.Parallel()

			a := make([]complex64, n)
			b := make([]complex64, n)
			want := make([]complex64, n)

			for i := range a {
				a[i] = complex(float32(i), float32(i+1))
				b[i] = complex(float32(i+2), float32(i+3))
				want[i] = a[i] * b[i]
			}

			dst := make([]complex64, n)
			ComplexMulArrayComplex64(dst, a, b)

			for i := range dst {
				if !complexNear64(dst[i], want[i], 1e-4) {
					t.Errorf("size %d: dst[%d] = %v, want %v", n, i, dst[i], want[i])
				}
			}
		})
	}
}

// Helper functions for comparing complex numbers with tolerance.
func complexNear64(a, b complex64, tol float32) bool {
	return float32(math.Abs(float64(real(a)-real(b)))) < tol &&
		float32(math.Abs(float64(imag(a)-imag(b)))) < tol
}

func complexNear128(a, b complex128, tol float64) bool {
	return cmplx.Abs(a-b) < tol
}
