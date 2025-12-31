package fft

import (
	"math"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// TestDITForward tests the ditForward generic wrapper
func TestDITForward(t *testing.T) {
	n := 8
	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	ok := ditForward(output, input, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("ditForward failed")
	}

	// DC component should be sum of inputs
	expected := complex64(complex(float32(n*(n-1)/2), 0))
	if diff := math.Abs(float64(real(output[0]) - real(expected))); diff > 1e-4 {
		t.Errorf("DC component mismatch: got %v, want %v", output[0], expected)
	}
}

// TestDITInverse tests the ditInverse generic wrapper
func TestDITInverse(t *testing.T) {
	n := 8
	freq := make([]complex64, n)
	freq[0] = complex64(complex(28, 0)) // Sum of 0..7

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	ok := ditInverse(output, freq, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("ditInverse failed")
	}

	// First element should be average (28/8 = 3.5)
	expected := float32(3.5)
	if diff := math.Abs(float64(real(output[0]) - expected)); diff > 1e-4 {
		t.Errorf("First element mismatch: got %v, want %v", output[0], expected)
	}
}

// TestStockhamForward tests the stockhamForward generic wrapper
func TestStockhamForward(t *testing.T) {
	n := 16
	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	ok := stockhamForward(output, input, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("stockhamForward failed")
	}

	// DC component should be sum of inputs
	expected := complex64(complex(float32(n*(n-1)/2), 0))
	if diff := math.Abs(float64(real(output[0]) - real(expected))); diff > 1e-4 {
		t.Errorf("DC component mismatch: got %v, want %v", output[0], expected)
	}
}

// TestStockhamInverse tests the stockhamInverse generic wrapper
func TestStockhamInverse(t *testing.T) {
	n := 16
	freq := make([]complex64, n)
	freq[0] = complex64(complex(120, 0)) // Sum of 0..15

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	ok := stockhamInverse(output, freq, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("stockhamInverse failed")
	}

	// First element should be average (120/16 = 7.5)
	expected := float32(7.5)
	if diff := math.Abs(float64(real(output[0]) - expected)); diff > 1e-4 {
		t.Errorf("First element mismatch: got %v, want %v", output[0], expected)
	}
}

// TestComputeChirpSequence tests chirp sequence computation for Bluestein's algorithm
func TestComputeChirpSequence(t *testing.T) {
	n := 7 // Non-power-of-2 size

	chirp := ComputeChirpSequence[complex64](n)
	if len(chirp) != n {
		t.Fatalf("chirp length = %d, want %d", len(chirp), n)
	}

	// Chirp sequence should have exp(-i*pi*k^2/n) pattern
	// First element chirp[0] = exp(0) = 1
	if diff := math.Abs(float64(real(chirp[0]) - 1.0)); diff > 1e-6 {
		t.Errorf("chirp[0] real = %v, want 1.0", real(chirp[0]))
	}
	if diff := math.Abs(float64(imag(chirp[0]))); diff > 1e-6 {
		t.Errorf("chirp[0] imag = %v, want 0.0", imag(chirp[0]))
	}

	// All elements should have magnitude 1
	for i, c := range chirp {
		mag := math.Sqrt(float64(real(c)*real(c) + imag(c)*imag(c)))
		if diff := math.Abs(mag - 1.0); diff > 1e-6 {
			t.Errorf("chirp[%d] magnitude = %v, want 1.0", i, mag)
		}
	}
}

// TestComputeBluesteinFilter tests Bluestein filter computation
func TestComputeBluesteinFilter(t *testing.T) {
	n := 7 // Non-power-of-2 size
	m := 16 // Next power of 2 >= 2n-1

	chirp := ComputeChirpSequence[complex64](n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](m)
	bitrev := mathpkg.ComputeBitReversalIndices(m)
	scratch := make([]complex64, m)

	filter := ComputeBluesteinFilter[complex64](n, m, chirp, twiddle, bitrev, scratch)
	if len(filter) != m {
		t.Fatalf("filter length = %d, want %d", len(filter), m)
	}

	// Filter should be non-zero
	hasNonZero := false
	for _, f := range filter {
		if real(f) != 0 || imag(f) != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("filter is all zeros")
	}
}

// TestBluesteinConvolution tests Bluestein convolution
func TestBluesteinConvolution(t *testing.T) {
	n := 7 // Non-power-of-2 size
	m := 16 // Next power of 2 >= 2n-1

	x := make([]complex64, m)
	for i := 0; i < n; i++ {
		x[i] = complex(float32(i), 0)
	}

	chirp := ComputeChirpSequence[complex64](n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](m)
	bitrev := mathpkg.ComputeBitReversalIndices(m)
	filter := ComputeBluesteinFilter[complex64](n, m, chirp, twiddle, bitrev, make([]complex64, m))

	scratch := make([]complex64, m)
	dst := make([]complex64, m)

	BluesteinConvolution(dst, x, filter, twiddle, scratch, bitrev)

	// Result should be non-zero
	hasNonZero := false
	for i := 0; i < n; i++ {
		if real(dst[i]) != 0 || imag(dst[i]) != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("convolution result is all zeros")
	}
}

// TestButterfly2 tests the butterfly2 wrapper
func TestButterfly2(t *testing.T) {
	a := complex64(complex(1, 2))
	b := complex64(complex(3, 4))
	w := complex64(complex(0.707, -0.707)) // exp(-i*pi/4)

	x, y := butterfly2(a, b, w)

	// Verify butterfly operation: x = a + w*b, y = a - w*b
	wb := complex64(complex(float64(real(w)*real(b)-imag(w)*imag(b)), float64(real(w)*imag(b)+imag(w)*real(b))))
	expectedX := a + wb
	expectedY := a - wb

	if diff := math.Abs(float64(real(x) - real(expectedX))); diff > 1e-4 {
		t.Errorf("butterfly2 x real mismatch: got %v, want %v", x, expectedX)
	}
	if diff := math.Abs(float64(imag(x) - imag(expectedX))); diff > 1e-4 {
		t.Errorf("butterfly2 x imag mismatch: got %v, want %v", x, expectedX)
	}
	if diff := math.Abs(float64(real(y) - real(expectedY))); diff > 1e-4 {
		t.Errorf("butterfly2 y real mismatch: got %v, want %v", y, expectedY)
	}
	if diff := math.Abs(float64(imag(y) - imag(expectedY))); diff > 1e-4 {
		t.Errorf("butterfly2 y imag mismatch: got %v, want %v", y, expectedY)
	}
}
