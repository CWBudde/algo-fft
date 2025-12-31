package fft

import (
	"math"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// TestForwardStridedDIT_Complex64 tests forward strided DIT FFT with complex64
func TestForwardStridedDIT_Complex64(t *testing.T) {
	n := 8
	stride := 2

	// Create input data with stride
	input := make([]complex64, n*stride)
	for i := 0; i < n; i++ {
		input[i*stride] = complex(float32(i), 0)
	}

	// Prepare twiddle factors and bit-reversal
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	// Output buffer
	output := make([]complex64, n*stride)

	// Run strided FFT
	ok := ForwardStridedDIT(output, input, twiddle, bitrev, stride, n)
	if !ok {
		t.Fatal("ForwardStridedDIT failed")
	}

	// Verify result - DC component should be sum of inputs
	expected := complex64(complex(float32(n*(n-1)/2), 0))
	if diff := math.Abs(float64(real(output[0]) - real(expected))); diff > 1e-4 {
		t.Errorf("DC component mismatch: got %v, want %v", output[0], expected)
	}
}

// TestInverseStridedDIT_Complex64 tests inverse strided DIT FFT with complex64
func TestInverseStridedDIT_Complex64(t *testing.T) {
	n := 8
	stride := 2

	// Create frequency domain data
	freq := make([]complex64, n*stride)
	freq[0] = complex64(complex(28, 0)) // Sum of 0..7

	// Prepare twiddle factors and bit-reversal
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	// Output buffer
	output := make([]complex64, n*stride)

	// Run inverse strided FFT
	ok := InverseStridedDIT(output, freq, twiddle, bitrev, stride, n)
	if !ok {
		t.Fatal("InverseStridedDIT failed")
	}

	// First element should be average (28/8 = 3.5)
	expected := float32(3.5)
	if diff := math.Abs(float64(real(output[0]) - expected)); diff > 1e-4 {
		t.Errorf("First element mismatch: got %v, want %v", output[0], expected)
	}
}

// TestStridedDIT_RoundTrip tests that inverse(forward(x)) = x
func TestStridedDIT_RoundTrip(t *testing.T) {
	n := 16
	stride := 3

	// Create input data
	input := make([]complex64, n*stride)
	for i := 0; i < n; i++ {
		input[i*stride] = complex(float32(i), float32(i*2))
	}

	// Prepare twiddle factors and bit-reversal
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	// Forward transform
	freq := make([]complex64, n*stride)
	ok := ForwardStridedDIT(freq, input, twiddle, bitrev, stride, n)
	if !ok {
		t.Fatal("ForwardStridedDIT failed")
	}

	// Inverse transform
	output := make([]complex64, n*stride)
	ok = InverseStridedDIT(output, freq, twiddle, bitrev, stride, n)
	if !ok {
		t.Fatal("InverseStridedDIT failed")
	}

	// Verify round-trip
	for i := 0; i < n; i++ {
		diff := math.Abs(float64(real(output[i*stride]) - real(input[i*stride])))
		if diff > 1e-4 {
			t.Errorf("Element %d real mismatch: got %v, want %v", i, output[i*stride], input[i*stride])
		}
		diff = math.Abs(float64(imag(output[i*stride]) - imag(input[i*stride])))
		if diff > 1e-4 {
			t.Errorf("Element %d imag mismatch: got %v, want %v", i, output[i*stride], input[i*stride])
		}
	}
}

// TestStridedDIT_ErrorHandling tests error conditions
func TestStridedDIT_ErrorHandling(t *testing.T) {
	n := 8
	stride := 2
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	tests := []struct {
		name   string
		dst    []complex64
		src    []complex64
		stride int
		n      int
		want   bool
	}{
		{
			name:   "zero size",
			dst:    make([]complex64, 16),
			src:    make([]complex64, 16),
			stride: stride,
			n:      0,
			want:   true, // zero size is valid
		},
		{
			name:   "invalid stride",
			dst:    make([]complex64, 16),
			src:    make([]complex64, 16),
			stride: 0,
			n:      n,
			want:   false,
		},
		{
			name:   "dst too small",
			dst:    make([]complex64, 10),
			src:    make([]complex64, 16),
			stride: stride,
			n:      n,
			want:   false,
		},
		{
			name:   "src too small",
			dst:    make([]complex64, 16),
			src:    make([]complex64, 10),
			stride: stride,
			n:      n,
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ForwardStridedDIT(tt.dst, tt.src, twiddle, bitrev, tt.stride, tt.n)
			if got != tt.want {
				t.Errorf("ForwardStridedDIT() = %v, want %v", got, tt.want)
			}
		})
	}
}
