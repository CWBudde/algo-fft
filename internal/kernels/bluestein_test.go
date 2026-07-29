package kernels

import (
	"math"
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
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

// TestBluesteinHelper drives ComputeBluesteinFilter and BluesteinConvolution
// through the full Bluestein assembly (pre-chirp, cyclic convolution,
// post-chirp) and compares the resulting spectrum bin-by-bin against the naive
// DFT.
//
// It used to assert only that the convolution output was not all zeros, which
// no wrong filter, wrong twiddle table or wrong output ordering could ever
// fail. The pre/post chirp multiplies live in internal/fft (which imports this
// package, so the test does them inline), but without them the convolution
// result is not comparable to anything and the test cannot check a value.
func TestBluesteinHelper(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ n, m int }{
		{3, 8},   // m = 8 >= 2*3-1
		{5, 16},  // odd n, larger pad
		{12, 32}, // composite n
	} {
		t.Run("n"+strconv.Itoa(tc.n), func(t *testing.T) {
			t.Parallel()

			n, m := tc.n, tc.m

			chirp := ComputeChirpSequence[complex128](n)
			twiddles := ComputeTwiddleFactors[complex128](m)
			scratch := make([]complex128, m)

			filter := ComputeBluesteinFilter(n, m, chirp, twiddles, scratch)
			if len(filter) != m {
				t.Fatalf("filter length mismatch: got %d, want %d", len(filter), m)
			}

			// Broadband input: every bin of the reference spectrum is
			// nonzero, so a wrong chirp, filter or bin ordering shows up.
			src := make([]complex128, n)
			for j := range src {
				src[j] = complex(math.Cos(0.7*float64(j))+0.25*float64(j),
					math.Sin(1.3*float64(j))-0.4)
			}

			// Pre-chirp into the zero-padded convolution input.
			x := make([]complex128, m)
			for j := range src {
				x[j] = src[j] * chirp[j]
			}

			dst := make([]complex128, m)
			BluesteinConvolution(dst, x, filter, twiddles, scratch, nil)

			// Post-chirp; only the first n samples are the spectrum.
			got := make([]complex128, n)
			for k := range got {
				got[k] = dst[k] * chirp[k]
			}

			want := reference.NaiveDFT128(src)
			for k := range want {
				if diff := cmplx.Abs(got[k] - want[k]); diff > 1e-10 {
					t.Errorf("n=%d bin %d: Bluestein %v, naive %v (diff %.3e)",
						n, k, got[k], want[k], diff)
				}
			}
		})
	}
}
