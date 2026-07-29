package transform

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// The companion tests in recursive_integration_test.go drive the recursive
// path with an impulse (x[0] = 1). That input is degenerate in exactly the two
// ways this path has already been wrong: its spectrum is all-ones, so every
// twiddle multiplies a zero and any permutation of the output is still
// all-ones. A wrong twiddle table and a wrong bin order both pass, as do
// Parseval and linearity — linearity holds for any linear operator, including
// the wrong one. That combination hid a wrong-spectrum bug in the recursive
// complex128 path at every size >= 1024 until a broadband signal was compared
// bin-by-bin against internal/reference.
//
// The tests here are that companion. Two input families:
//
//   - a broadband signal (incommensurate tones plus a ramp, different phase in
//     the imaginary part) compared against the naive DFT, for sizes where the
//     O(n^2) reference is still cheap;
//   - a sum of complex exponentials at distinct bins, whose spectrum is known
//     in closed form (n*amplitude at those bins, zero elsewhere), for the
//     larger sizes. It costs O(n) instead of O(n^2) and is not blind: a wrong
//     twiddle leaks energy into the zero bins, a wrong ordering puts the
//     spikes in the wrong ones.

// recursiveBroadbandCodeletSizes mirrors the codelet ladder the sibling tests
// use, so the same decompositions are exercised.
//
//nolint:gochecknoglobals
var recursiveBroadbandCodeletSizes = []int{4, 8, 16, 32, 64, 128, 256, 512}

const recursiveBroadbandCache = 32768 // 32 KB L1

// referenceSweepSizes are the sizes compared against the naive DFT. 8192 and
// 16384 are left to the multi-tone test: the reference costs 1.1 s and 4.3 s
// there, which is not worth paying on every run (and far worse under -race).
//
//nolint:gochecknoglobals
var referenceSweepSizes = []int{512, 1024, 2048, 4096}

// multiToneSizes are the sizes covered by the closed-form spectrum instead.
//
//nolint:gochecknoglobals
var multiToneSizes = []int{1024, 8192, 16384}

// broadbandSignal64 has no zero bin in its spectrum and no symmetry a
// permutation could preserve.
func broadbandSignal64(n int) []complex64 {
	src := make([]complex64, n)
	for i := range src {
		f := float64(i)
		src[i] = complex(
			float32(math.Cos(0.7*f)+0.3*math.Sin(2.9*f)+0.05*math.Sqrt(f)),
			float32(math.Sin(1.3*f)-0.4*math.Cos(0.11*f)),
		)
	}

	return src
}

// broadbandSignal128 is broadbandSignal64 at the wider precision.
func broadbandSignal128(n int) []complex128 {
	src := make([]complex128, n)
	for i := range src {
		f := float64(i)
		src[i] = complex(
			math.Cos(0.7*f)+0.3*math.Sin(2.9*f)+0.05*math.Sqrt(f),
			math.Sin(1.3*f)-0.4*math.Cos(0.11*f),
		)
	}

	return src
}

// toneSpec is one component of a multi-tone signal: amplitude a at bin k.
type toneSpec struct {
	k int
	a complex128
}

// multiTones picks bins spread across the spectrum, including ones a
// bit-reversal or a decimation-order error would move somewhere visible.
func multiTones(n int) []toneSpec {
	return []toneSpec{
		{1, complex(1, 0)},
		{3, complex(0, -2)},
		{n/4 + 1, complex(0.5, 0.25)},
		{n / 2, complex(-1.5, 0)},
		{n - 7, complex(0.75, 1.25)},
	}
}

// multiToneSignal128 evaluates sum_t a_t * exp(2*pi*i*k_t*j/n).
func multiToneSignal128(n int, tones []toneSpec) []complex128 {
	src := make([]complex128, n)

	for _, t := range tones {
		for j := range src {
			// k*j is reduced modulo n before scaling to radians: the unreduced
			// angle reaches ~1e5 rad at these sizes, and the phase error from
			// argument reduction alone would then dominate the tolerance.
			phase := 2 * math.Pi * float64((int64(t.k)*int64(j))%int64(n)) / float64(n)
			src[j] += t.a * cmplx.Exp(complex(0, phase))
		}
	}

	return src
}

// multiToneSpectrum128 is the exact forward DFT of multiToneSignal128 under
// the library's unnormalized forward convention.
func multiToneSpectrum128(n int, tones []toneSpec) []complex128 {
	want := make([]complex128, n)
	for _, t := range tones {
		want[t.k] += t.a * complex(float64(n), 0)
	}

	return want
}

// runRecursiveForward64 runs one recursive forward transform at the given size.
func runRecursiveForward64(t *testing.T, n int, src []complex64) []complex64 {
	t.Helper()

	strategy := PlanDecomposition(n, recursiveBroadbandCodeletSizes, recursiveBroadbandCache)
	dst := make([]complex64, n)
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	scratch := make([]complex64, ScratchSizeRecursive(strategy))

	recursiveForward(dst, src, strategy, twiddle, scratch, registry.Registry64, cpu.DetectFeatures())

	return dst
}

// runRecursiveForward128 is runRecursiveForward64 at the wider precision.
func runRecursiveForward128(t *testing.T, n int, src []complex128) []complex128 {
	t.Helper()

	strategy := PlanDecomposition(n, recursiveBroadbandCodeletSizes, recursiveBroadbandCache)
	dst := make([]complex128, n)
	twiddle := TwiddleFactorsRecursive[complex128](strategy)
	scratch := make([]complex128, ScratchSizeRecursive(strategy))

	recursiveForward(dst, src, strategy, twiddle, scratch, registry.Registry128, cpu.DetectFeatures())

	return dst
}

// peakAbs128 is the largest magnitude in a spectrum, used to scale tolerances:
// bounding each bin against its own magnitude sets an unreachable target for
// the near-zero ones.
func peakAbs128(x []complex128) float64 {
	peak := 0.0
	for _, v := range x {
		peak = math.Max(peak, cmplx.Abs(v))
	}

	return peak
}

// TestRecursiveForwardBroadbandComplex64 is the broadband companion to
// TestRecursiveFFTCorrectness. The reference is accumulated in float64
// (NaiveDFTWide) so the comparison is not limited by the reference's own
// float32 rounding.
func TestRecursiveForwardBroadbandComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range referenceSweepSizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			src := broadbandSignal64(n)
			got := runRecursiveForward64(t, n, src)
			want := reference.NaiveDFTWide(src)

			tol := 1e-5 * peakAbs128(want)
			for i := range got {
				if diff := cmplx.Abs(complex128(got[i]) - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: got %v, want %v (diff %.3e > %.3e)",
						n, i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

// TestRecursiveForwardBroadbandComplex128 is the same sweep at the precision
// that carried the original bug. TestRecursiveFFTComplex128 covers one size
// with a ramp; this covers the whole ladder with a signal that has energy in
// every bin.
func TestRecursiveForwardBroadbandComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range referenceSweepSizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			src := broadbandSignal128(n)
			got := runRecursiveForward128(t, n, src)
			want := reference.NaiveDFT128(src)

			tol := 1e-12 * peakAbs128(want)
			for i := range got {
				if diff := cmplx.Abs(got[i] - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: got %v, want %v (diff %.3e > %.3e)",
						n, i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

// TestRecursiveForwardMultiToneComplex64 covers the sizes the O(n^2) reference
// is too slow for. The expected spectrum is exact, so both the spike positions
// (ordering) and the empty bins (twiddle leakage) are checked.
func TestRecursiveForwardMultiToneComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range multiToneSizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			tones := multiTones(n)

			wide := multiToneSignal128(n, tones)
			src := make([]complex64, n)

			for i, v := range wide {
				src[i] = complex64(v)
			}

			got := runRecursiveForward64(t, n, src)
			want := multiToneSpectrum128(n, tones)

			// The tolerance is set by the float32 input itself: rounding the
			// signal to complex64 perturbs every bin by ~n*eps32 already,
			// before the transform adds its own log(n) stages of rounding.
			// Measured headroom at 16384 is ~10x.
			tol := 3e-7 * float64(n)
			for i := range got {
				if diff := cmplx.Abs(complex128(got[i]) - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: got %v, want %v (diff %.3e > %.3e)",
						n, i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

// TestRecursiveForwardMultiToneComplex128 is the complex128 twin, covering the
// >= 1024 complex128 range where the recursive path was silently wrong.
func TestRecursiveForwardMultiToneComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range multiToneSizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			tones := multiTones(n)
			src := multiToneSignal128(n, tones)
			got := runRecursiveForward128(t, n, src)
			want := multiToneSpectrum128(n, tones)

			// Measured headroom at 16384 is ~80x.
			tol := 5e-15 * float64(n)
			for i := range got {
				if diff := cmplx.Abs(got[i] - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: got %v, want %v (diff %.3e > %.3e)",
						n, i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

// TestRecursiveRoundTripBroadband replaces the impulse in the round-trip check
// with a broadband signal. Inverse(Forward(x)) == x is a cheap self-consistency
// check that needs no reference, but on an impulse it too passes trivially.
func TestRecursiveRoundTripBroadband(t *testing.T) {
	t.Parallel()

	for _, n := range referenceSweepSizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(n, recursiveBroadbandCodeletSizes, recursiveBroadbandCache)
			features := cpu.DetectFeatures()

			src := broadbandSignal128(n)
			fwd := make([]complex128, n)
			back := make([]complex128, n)
			twiddle := TwiddleFactorsRecursive[complex128](strategy)
			scratch := make([]complex128, ScratchSizeRecursive(strategy))

			recursiveForward(fwd, src, strategy, twiddle, scratch, registry.Registry128, features)
			recursiveInverse(back, fwd, strategy, twiddle, scratch, registry.Registry128, features)

			tol := 1e-12 * peakAbs128(src)
			for i := range back {
				if diff := cmplx.Abs(back[i] - src[i]); diff > tol {
					t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v (diff %.3e > %.3e)",
						n, i, back[i], src[i], diff, tol)
				}
			}
		})
	}
}
