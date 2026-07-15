package algofft

import (
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestBluesteinPadSize checks the plan-time pad chooser. With the current
// calibration (see bluesteinSubFFTPenalty) the size-dispatched power-of-two
// sub-FFT wins at every size, so the chooser must return the next power of
// two >= 2n-1 — while still satisfying the structural invariants that hold
// for any calibration.
func TestBluesteinPadSize(t *testing.T) {
	t.Parallel()

	for _, n := range []int{2, 7, 11, 13, 127, 251, 257, 499, 509, 677, 997, 1009, 1021, 2531, 4099} {
		got := bluesteinPadSize(n)

		if got < 2*n-1 {
			t.Errorf("bluesteinPadSize(%d) = %d < 2n-1 = %d", n, got, 2*n-1)
		}

		if !m.IsHighlyComposite(got) {
			t.Errorf("bluesteinPadSize(%d) = %d is not executable by the mixed-radix engine", n, got)
		}

		if want := m.NextPowerOfTwo(2*n - 1); got != want {
			t.Errorf("bluesteinPadSize(%d) = %d, want %d (current calibration always picks the power of two)",
				n, got, want)
		}
	}
}

// TestBluestein_SmoothPadMatchesReference exercises the 5-smooth padded
// Bluestein machinery end to end: table construction with a non-power-of-two
// m (mixed-radix filter FFT, no bitrev table) and the mixed-radix convolution
// path, validated against the naive DFT. The plan-time chooser currently
// always picks the power of two, so this path is driven directly.
func TestBluestein_SmoothPadMatchesReference(t *testing.T) {
	t.Parallel()

	// Prime n with a 5-smooth pad well below the next power of two.
	cases := []struct{ n, m int }{
		{n: 13, m: 25},     // 5^2
		{n: 257, m: 540},   // 2^2·3^3·5 (pow2 would be 1024)
		{n: 1009, m: 2025}, // 3^4·5^2 (pow2 would be 2048)
	}

	for _, tc := range cases {
		if tc.m < 2*tc.n-1 || m.IsPowerOf2(tc.m) || !m.IsHighlyComposite(tc.m) {
			t.Fatalf("bad test case n=%d m=%d", tc.n, tc.m)
		}

		t.Run("complex128_"+itoa(tc.n), func(t *testing.T) {
			t.Parallel()

			scratch := make([]complex128, tc.m)
			chirp, chirpInv, filter, filterInv, twiddle, bitrev := computeBluesteinTables[complex128](tc.n, tc.m, scratch)

			if bitrev != nil {
				t.Errorf("bitrev table computed for non-power-of-two m=%d", tc.m)
			}

			src := make([]complex128, tc.n)
			for i := range src {
				src[i] = complex(float64(i%17)-8, float64((i*i)%23)-11)
			}

			// Forward: dst[k] = chirp[k] · IFFT(FFT(chirp·src) · filter)[k].
			work := make([]complex128, tc.m)
			for i := range src {
				work[i] = src[i] * chirp[i]
			}

			bsScratch := make([]complex128, tc.m)
			fft.BluesteinConvolution(work, work, filter, twiddle, bsScratch, bitrev)

			dst := make([]complex128, tc.n)
			for i := range dst {
				dst[i] = work[i] * chirp[i]
			}

			ref := reference.NaiveDFT128(src)
			for i := range dst {
				if diff := cmplx.Abs(dst[i] - ref[i]); diff > 1e-9 {
					t.Fatalf("forward bin %d: got %v, want %v (diff %g)", i, dst[i], ref[i], diff)
				}
			}

			// Inverse: x[k] = chirpInv[k]/n · IFFT(FFT(chirpInv·X) · filterInv)[k].
			for i := range work {
				work[i] = 0
			}

			for i := range dst {
				work[i] = dst[i] * chirpInv[i]
			}

			fft.BluesteinConvolution(work, work, filterInv, twiddle, bsScratch, bitrev)

			scale := complex(1.0/float64(tc.n), 0)

			for i := range src {
				got := work[i] * chirpInv[i] * scale
				if diff := cmplx.Abs(got - src[i]); diff > 1e-9 {
					t.Fatalf("inverse bin %d: got %v, want %v (diff %g)", i, got, src[i], diff)
				}
			}
		})
	}
}

// TestBluestein_LargePrimesMatchReference validates the (size-dispatched)
// power-of-two Bluestein sub-FFT against the naive DFT at padded sizes that
// engage the optimized kernels (m = 1024, 2048) and the generic fallback
// above the dispatch bound (m = 16384). The pre-existing reference tests
// stop at n=31 (m=64); the round-trip-only large-prime tests would not catch
// a systematically wrong spectrum.
func TestBluestein_LargePrimesMatchReference(t *testing.T) {
	t.Parallel()

	for _, n := range []int{509, 1021, 4099} {
		// Shared complex128 reference; the naive complex64 DFT accumulates too
		// much rounding error of its own at these sizes to serve as ground truth.
		src128 := make([]complex128, n)
		for i := range src128 {
			src128[i] = complex(float64(i%17)-8, float64((i*i)%23)-11)
		}

		ref := reference.NaiveDFT128(src128)

		t.Run("complex64_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanT[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex64(src128[i])
			}

			dst := make([]complex64, n)

			err = plan.Forward(dst, src)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			for i := range dst {
				rel := cmplx.Abs(complex128(dst[i])-ref[i]) / (cmplx.Abs(ref[i]) + 1)
				if rel > 1e-3 {
					t.Fatalf("bin %d: got %v, want %v (rel %g)", i, dst[i], ref[i], rel)
				}
			}
		})

		t.Run("complex128_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanT[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			dst := make([]complex128, n)

			err = plan.Forward(dst, src128)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			for i := range dst {
				rel := cmplx.Abs(dst[i]-ref[i]) / (cmplx.Abs(ref[i]) + 1)
				if rel > 1e-9 {
					t.Fatalf("bin %d: got %v, want %v (rel %g)", i, dst[i], ref[i], rel)
				}
			}
		})
	}
}
