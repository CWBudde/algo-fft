package kernels

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestComputeChirpSequence(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n int
	}{
		{4},
		{8},
		{3},
		{5},
	}

	for _, tt := range tests {
		t.Run("complex64", func(t *testing.T) {
			t.Parallel()

			chirp := ComputeChirpSequence[complex64](tt.n)
			if len(chirp) != tt.n {
				t.Errorf("expected length %d, got %d", tt.n, len(chirp))
			}

			validateChirp(t, chirp, tt.n)
		})
		t.Run("complex128", func(t *testing.T) {
			t.Parallel()

			chirp := ComputeChirpSequence[complex128](tt.n)
			if len(chirp) != tt.n {
				t.Errorf("expected length %d, got %d", tt.n, len(chirp))
			}

			validateChirp(t, chirp, tt.n)
		})
	}
}

func validateChirp[T Complex](t *testing.T, chirp []T, n int) {
	t.Helper()

	// Check w_0 = 1
	if cmplx.Abs(complex128(chirp[0])-1) > 1e-6 {
		t.Errorf("w_0 should be 1, got %v", chirp[0])
	}

	// Check w_k = exp(-j * pi * k^2 / n)
	for k := range n {
		angle := -math.Pi * float64(k*k) / float64(n)
		expected := cmplx.Rect(1, angle)

		got := complex128(chirp[k])
		if cmplx.Abs(got-expected) > 1e-5 {
			t.Errorf("w_%d: expected %v, got %v", k, expected, got)
		}
	}

	// Check symmetry
	// if N is even: w_{N-k} = w_k
	// if N is odd: w_{N-k} = -w_k (Wait, let's re-verify)
	// My previous derivation:
	// w_{N-k} = (-1)^N * w_k
	// For k=1..N-1
	for k := 1; k < n; k++ {
		val := complex128(chirp[k])
		mirror := complex128(chirp[n-k])

		var expectedMirror complex128
		if n%2 == 0 {
			expectedMirror = val
		} else {
			expectedMirror = -val
		}

		if cmplx.Abs(mirror-expectedMirror) > 1e-5 {
			t.Errorf("Symmetry check failed for k=%d: expected %v, got %v", k, expectedMirror, mirror)
		}
	}
}

func TestBluesteinHelper(t *testing.T) {
	t.Parallel()

	// Simple test for ComputeBluesteinFilter and BluesteinConvolution
	// We won't verify the full convolution result correctness here rigorously (that's for integration tests),
	// but we'll check that it runs and produces output.
	n := 3
	m := 8 // Power of 2 >= 2*3-1 = 5

	chirp := ComputeChirpSequence[complex128](n)
	twiddles := ComputeTwiddleFactors[complex128](m)

	scratch := make([]complex128, m)

	filter := ComputeBluesteinFilter(n, m, chirp, twiddles, scratch)
	if len(filter) != m {
		t.Errorf("Filter length mismatch: got %d, want %d", len(filter), m)
	}

	x := make([]complex128, m)
	x[0] = 1
	x[1] = 2
	x[2] = 3

	dst := make([]complex128, m)

	BluesteinConvolution(dst, x, filter, twiddles, scratch)

	// Basic check: output should not be all zeros (unless inputs determine so, which they don't)
	allZero := true

	for _, v := range dst {
		if v != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Errorf("Convolution output is all zeros")
	}
}
