package algofft

import (
	"errors"
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestBluesteinPadSize checks the plan-time pad chooser against the calibrated
// shape whitelist (see padShapes): 3*2^(k-2) is taken once the power-of-two pad
// reaches 2^9 and 15*2^(k-4) once it reaches 2^13, each only when it still
// covers the required 2n-1; otherwise the power of two stands.
func TestBluesteinPadSize(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want int
	}{
		{n: 2, want: 4},          // pad 2^2, below the 3*2^(k-2) threshold
		{n: 7, want: 16},         // pad 2^4, below the threshold
		{n: 11, want: 32},        // pad 2^5, below the threshold
		{n: 13, want: 32},        // pad 2^5, below the threshold
		{n: 127, want: 256},      // pad 2^8, one window below the threshold
		{n: 251, want: 512},      // pad 2^9: 384 < 2n-1 = 501, power of two stands
		{n: 257, want: 768},      // pad 2^10: 768 >= 513
		{n: 499, want: 1024},     // pad 2^10: 768 < 997
		{n: 509, want: 1024},     // pad 2^10: 768 < 1017
		{n: 677, want: 1536},     // pad 2^11: 1536 >= 1353
		{n: 997, want: 2048},     // pad 2^11: 1536 < 1993
		{n: 1009, want: 2048},    // pad 2^11: 1536 < 2017
		{n: 1021, want: 2048},    // pad 2^11: 1536 < 2041
		{n: 2531, want: 6144},    // pad 2^13: 6144 >= 5061
		{n: 3079, want: 7680},    // pad 2^13: 6144 < 6157, 15*2^(k-4) = 7680 covers it
		{n: 4001, want: 8192},    // pad 2^13: 7680 < 8001, power of two stands
		{n: 4099, want: 12288},   // pad 2^14: 12288 >= 8197
		{n: 6151, want: 15360},   // pad 2^14: 12288 < 12301, 15360 covers it
		{n: 65537, want: 196608}, // pad 2^18: 3*2^16 >= 131073
	}

	for _, tt := range tests {
		got := bluesteinPadSize(tt.n)

		if got < 2*tt.n-1 {
			t.Errorf("bluesteinPadSize(%d) = %d < 2n-1 = %d", tt.n, got, 2*tt.n-1)
		}

		if !m.IsHighlyComposite(got) {
			t.Errorf("bluesteinPadSize(%d) = %d is not executable by the mixed-radix engine", tt.n, got)
		}

		if got != tt.want {
			t.Errorf("bluesteinPadSize(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

// TestBluesteinPadSize_Invariants sweeps every length up to a few thousand and
// pins the properties that must hold whatever the calibration says: the pad
// covers the cyclic convolution, never exceeds the power of two it replaces,
// and is a length both the raw mixed-radix engine and the planner accept — the
// latter because fastConvolutionLength turns the same value into a plan length.
func TestBluesteinPadSize_Invariants(t *testing.T) {
	t.Parallel()

	for n := 2; n <= 5000; n++ {
		got := bluesteinPadSize(n)
		minM := 2*n - 1

		if got < minM {
			t.Fatalf("bluesteinPadSize(%d) = %d < 2n-1 = %d", n, got, minM)
		}

		if pow2 := m.NextPowerOfTwo(minM); got > pow2 {
			t.Fatalf("bluesteinPadSize(%d) = %d > next power of two %d", n, got, pow2)
		}

		if !m.IsMixedRadixSmooth(got) {
			t.Fatalf("bluesteinPadSize(%d) = %d is not mixed-radix executable", n, got)
		}

		if !m.IsPowerOf2(got) && !planner.MixedRadixEligible(got) {
			t.Fatalf("bluesteinPadSize(%d) = %d would not be routed to the mixed-radix engine", n, got)
		}
	}
}

// TestNewPlan_BluesteinTooLarge pins the plan-boundary guard: lengths whose
// Bluestein pad size (>= 2n-1) would overflow int are rejected with
// ErrInvalidLength instead of planning against wrapped arithmetic.
// maxBluesteinLength+1 is divisible by 3 but not 5-smooth, so it resolves to
// the Bluestein strategy on both 32- and 64-bit platforms.
func TestNewPlan_BluesteinTooLarge(t *testing.T) {
	t.Parallel()

	n := maxBluesteinLength + 1

	if m.IsPowerOf2(n) || m.IsHighlyComposite(n) {
		t.Fatalf("test premise broken: %d would not use Bluestein", n)
	}

	_, err := NewPlan[complex64](n)
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("NewPlan[complex64](%d) error = %v, want ErrInvalidLength", n, err)
	}

	_, err = NewPlan[complex128](n)
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("NewPlan[complex128](%d) error = %v, want ErrInvalidLength", n, err)
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

		t.Run("complex128_"+strconv.Itoa(tc.n), func(t *testing.T) {
			t.Parallel()

			scratch := make([]complex128, tc.m)
			probe := make([]complex128, tc.m)
			tables := computeBluesteinTables[complex128](tc.n, tc.m, cpu.DetectFeatures(), probe, scratch)
			chirp, chirpInv := tables.chirp, tables.chirpInv
			filter, filterInv := tables.filter, tables.filterInv
			twiddle, bitrev := tables.twiddle, tables.bitrev

			if bitrev != nil {
				t.Errorf("bitrev table computed for non-power-of-two m=%d", tc.m)
			}

			// Non-power-of-two pads run the mixed-radix engine, which is
			// already SIMD-dispatched and takes no bound sub-FFT.
			if tables.sub != nil {
				t.Errorf("sub-FFT bound for non-power-of-two m=%d", tc.m)
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
			fft.BluesteinConvolution(work, work, filter, twiddle, bsScratch, bitrev, tables.sub)

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

			fft.BluesteinConvolution(work, work, filterInv, twiddle, bsScratch, bitrev, tables.sub)

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

// TestBluestein_LargePrimesMatchReference validates the Bluestein sub-FFT
// against the naive DFT at padded sizes that engage the optimized kernels
// (m = 1024, 2048) and at the mixed-radix padded sizes the shape-aware pad
// model selects. The pre-existing reference tests stop at n=31 (m=64); the
// round-trip-only large-prime tests would not catch a systematically wrong
// spectrum.
func TestBluestein_LargePrimesMatchReference(t *testing.T) {
	t.Parallel()

	// 509 and 1021 keep a power-of-two pad; 2531 pads to 6144 = 2^11·3 and
	// 4099 to 12288 = 2^12·3, so both directions of the mixed-radix padded
	// path are checked against the reference spectrum.
	for _, n := range []int{509, 1021, 2531, 4099} {
		// Shared complex128 reference; the naive complex64 DFT accumulates too
		// much rounding error of its own at these sizes to serve as ground truth.
		src128 := make([]complex128, n)
		for i := range src128 {
			src128[i] = complex(float64(i%17)-8, float64((i*i)%23)-11)
		}

		ref := reference.NaiveDFT128(src128)

		t.Run("complex64_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
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

		t.Run("complex128_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
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
