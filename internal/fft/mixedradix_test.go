package fft

import (
	"math"
	"math/cmplx"
	"strconv"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// mixedRadixBroadbandSizes covers every radix the scheduler emits (2/3/4/5/7/8
// /11) and several stage orders of each, at lengths where the naive reference
// is still cheap.
//
// 1155 and 1331 are deliberately absent: measure_codelet_test.go registers
// identity stub codelets at those sizes, and the registry is global and
// append-only, so any transform at them silently returns its input for the rest
// of the package's tests. See the warning there.
//
//nolint:gochecknoglobals
var mixedRadixBroadbandSizes = []int{
	12, 15, 20, 24, 45, 60, 63, 96, 105, 176, 231, 240, 385, 720, 1260,
}

// broadbandComplex64 builds a signal with no zero bins in its spectrum and no
// symmetry a permutation could preserve: two incommensurate tones plus a ramp,
// with a different phase in the imaginary part. A degenerate input (impulse,
// constant, single lattice bin) cannot distinguish a wrong twiddle table or a
// wrong output ordering from a right one.
func broadbandComplex64(n int) []complex64 {
	src := make([]complex64, n)
	for i := range src {
		f := float64(i)
		src[i] = complex(
			float32(math.Cos(0.7*f)+0.3*math.Sin(2.9*f)+0.05*f),
			float32(math.Sin(1.3*f)-0.4*math.Cos(0.11*f)),
		)
	}

	return src
}

// broadbandComplex128 is broadbandComplex64 at the wider precision.
func broadbandComplex128(n int) []complex128 {
	src := make([]complex128, n)
	for i := range src {
		f := float64(i)
		src[i] = complex(
			math.Cos(0.7*f)+0.3*math.Sin(2.9*f)+0.05*f,
			math.Sin(1.3*f)-0.4*math.Cos(0.11*f),
		)
	}

	return src
}

// TestMixedRadixBroadbandComplex64 drives the mixed-radix engine with a
// broadband signal across every radix in the schedule and compares bin-by-bin
// against the naive DFT accumulated in float64 (NaiveDFTWide), so the reference
// is not itself limited by float32 rounding.
func TestMixedRadixBroadbandComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range mixedRadixBroadbandSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			src := broadbandComplex64(n)
			twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
			scratch := make([]complex64, n*2)
			dst := make([]complex64, n)

			if !forwardMixedRadixComplex64(dst, src, twiddle, scratch) {
				t.Fatalf("n=%d: forwardMixedRadixComplex64 declined", n)
			}

			want := reference.NaiveDFTWide(src)

			var peak float64
			for _, v := range want {
				peak = math.Max(peak, cmplx.Abs(v))
			}

			// Bound relative to the spectrum peak: near-zero bins otherwise
			// set a target float32 accumulation cannot reach.
			tol := 1e-5 * peak
			for i := range dst {
				if diff := cmplx.Abs(complex128(dst[i]) - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: got %v, want %v (diff %.3e > %.3e)",
						n, i, dst[i], want[i], diff, tol)
				}
			}

			// Round-trip on the same broadband signal.
			fwd := make([]complex64, n)
			copy(fwd, dst)

			if !inverseMixedRadixComplex64(dst, fwd, twiddle, scratch) {
				t.Fatalf("n=%d: inverseMixedRadixComplex64 declined", n)
			}

			var srcPeak float64
			for _, v := range src {
				srcPeak = math.Max(srcPeak, cmplx.Abs(complex128(v)))
			}

			for i := range dst {
				if diff := cmplx.Abs(complex128(dst[i] - src[i])); diff > 1e-5*srcPeak {
					t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, dst[i], src[i])
				}
			}
		})
	}
}

// TestMixedRadixBroadbandComplex128 is the complex128 twin. The recursive path
// carried a wrong-spectrum bug for an entire precision once already, so both
// precisions get the same broadband sweep rather than one sharing the other's
// coverage.
func TestMixedRadixBroadbandComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range mixedRadixBroadbandSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			src := broadbandComplex128(n)
			twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
			scratch := make([]complex128, n*2)
			dst := make([]complex128, n)

			if !forwardMixedRadixComplex128(dst, src, twiddle, scratch) {
				t.Fatalf("n=%d: forwardMixedRadixComplex128 declined", n)
			}

			want := reference.NaiveDFT128(src)

			var peak float64
			for _, v := range want {
				peak = math.Max(peak, cmplx.Abs(v))
			}

			tol := 1e-12 * peak
			for i := range dst {
				if diff := cmplx.Abs(dst[i] - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: got %v, want %v (diff %.3e > %.3e)",
						n, i, dst[i], want[i], diff, tol)
				}
			}

			fwd := make([]complex128, n)
			copy(fwd, dst)

			if !inverseMixedRadixComplex128(dst, fwd, twiddle, scratch) {
				t.Fatalf("n=%d: inverseMixedRadixComplex128 declined", n)
			}

			var srcPeak float64
			for _, v := range src {
				srcPeak = math.Max(srcPeak, cmplx.Abs(v))
			}

			for i := range dst {
				if diff := cmplx.Abs(dst[i] - src[i]); diff > 1e-12*srcPeak {
					t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, dst[i], src[i])
				}
			}
		})
	}
}

func TestMixedRadixComplex64(t *testing.T) {
	t.Parallel()

	// 12 = 4 * 3 (mixed radix)
	n := 12

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n*2) // Extra scratch for recursive
	dst := make([]complex64, n)

	// Forward
	if !forwardMixedRadixComplex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardMixedRadixComplex64 failed")
	}

	ref := reference.NaiveDFT(src)
	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-ref[i])) > 1e-5 {
			t.Errorf("forwardMixedRadixComplex64 mismatch at %d: got %v want %v", i, dst[i], ref[i])
		}
	}

	// Inverse
	fwd := make([]complex64, n)
	copy(fwd, dst)

	if !inverseMixedRadixComplex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseMixedRadixComplex64 failed")
	}

	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-src[i])) > 1e-5 {
			t.Errorf("inverseMixedRadixComplex64 mismatch at %d", i)
		}
	}
}

func TestMixedRadixComplex128(t *testing.T) {
	t.Parallel()

	n := 12

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n*2)
	dst := make([]complex128, n)

	// Forward
	if !forwardMixedRadixComplex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardMixedRadixComplex128 failed")
	}

	ref := reference.NaiveDFT128(src)
	for i := range dst {
		if cmplx.Abs(dst[i]-ref[i]) > 1e-10 {
			t.Errorf("forwardMixedRadixComplex128 mismatch at %d", i)
		}
	}

	// Inverse
	fwd := make([]complex128, n)
	copy(fwd, dst)

	if !inverseMixedRadixComplex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseMixedRadixComplex128 failed")
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-src[i]) > 1e-10 {
			t.Errorf("inverseMixedRadixComplex128 mismatch at %d", i)
		}
	}
}
