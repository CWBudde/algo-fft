package algofft

import (
	"math/cmplx"
	"math/rand"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// raderTestPrimes are prime lengths the planner upgrades from Bluestein to
// Rader's algorithm (the mixed-radix engine executes n-1 exactly and the
// shape is a measured win; see fft.RaderEligible). 17/257 exercise the
// power-of-two sub-FFT path, 97..1601 the 5-smooth mixed-radix path with a
// codelet leaf (97 -> [3, 32], 1153 -> [3, 3, 128], ...), and 113..2269 the
// radix-7/11 stages (113 -> [7, 16], 353 -> [11, 32], 2269 -> [7, 4, 3, 3,
// 3, 3]).
var raderTestPrimes = []int{
	17, 97, 257, 401, 641, 769, 1153, 1601,
	113, 353, 449, 673, 1409, 2113, 2269,
}

// raderFallbackPrimes stay on Bluestein: either the mixed-radix engine cannot
// execute n-1 at all (47: 46 = 2*23) or the exact sub-FFT measured slower
// than the padded one — power-of-two part of n-1 <= 4 (31, 101, 151, 251,
// 23, 29, 127, 463, 2311), or a radix-7/11 schedule below 2048 that is too
// shallow (281, 1321) or too odd-heavy (881, 1009, 2017).
var raderFallbackPrimes = []int{
	23, 29, 31, 47, 101, 127, 151, 251, 281, 463, 881, 1009, 1321, 2017, 2311,
}

func randomComplex64(n int, seed int64) []complex64 {
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test data

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(rng.Float64()*2-1), float32(rng.Float64()*2-1))
	}

	return src
}

func randomComplex128(n int, seed int64) []complex128 {
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test data

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	return src
}

func TestRader_PlanSelection(t *testing.T) {
	t.Parallel()

	for _, n := range raderTestPrimes {
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		if plan.Algorithm() != algorithmRader {
			t.Errorf("n=%d: Algorithm() = %q, want \"rader\"", n, plan.Algorithm())
		}

		if plan.KernelStrategy() != KernelBluestein {
			t.Errorf("n=%d: strategy = %v, want KernelBluestein", n, plan.KernelStrategy())
		}
	}

	for _, n := range raderFallbackPrimes {
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		if plan.Algorithm() == algorithmRader {
			t.Errorf("n=%d: Algorithm() = \"rader\", want Bluestein fallback", n)
		}
	}
}

// TestRader_ForcedBluestein verifies an explicitly forced KernelBluestein is
// honored even for Rader-eligible sizes.
func TestRader_ForcedBluestein(t *testing.T) {
	t.Parallel()

	n := 257

	plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelBluestein})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(%d) failed: %v", n, err)
	}

	if plan.Algorithm() == algorithmRader {
		t.Fatalf("forced KernelBluestein still selected Rader")
	}

	src := randomComplex64(n, 1)
	dst := make([]complex64, n)

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	want := reference.NaiveDFT(src)
	for i := range dst {
		if diff := cmplx.Abs(complex128(dst[i] - want[i])); diff > 1e-2 {
			t.Fatalf("bin %d: FFT=%v, naive=%v, diff=%v", i, dst[i], want[i], diff)
		}
	}
}

// TestRader_MatchesReference validates the Rader path against the naive DFT.
func TestRader_MatchesReference(t *testing.T) {
	t.Parallel()

	for _, n := range raderTestPrimes {
		t.Run("complex64_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
			}

			src := randomComplex64(n, int64(n))
			dst := make([]complex64, n)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			want := reference.NaiveDFT(src)
			for i := range dst {
				if diff := cmplx.Abs(complex128(dst[i] - want[i])); diff > 1e-2 {
					t.Errorf("bin %d: FFT=%v, naive=%v, diff=%v", i, dst[i], want[i], diff)
				}
			}
		})

		t.Run("complex128_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
			}

			src := randomComplex128(n, int64(n))
			dst := make([]complex128, n)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			want := reference.NaiveDFT128(src)
			for i := range dst {
				if diff := cmplx.Abs(dst[i] - want[i]); diff > 1e-9 {
					t.Errorf("bin %d: FFT=%v, naive=%v, diff=%v", i, dst[i], want[i], diff)
				}
			}
		})
	}
}

// TestRader_InverseMatchesReference validates the Rader inverse against the
// naive IDFT.
func TestRader_InverseMatchesReference(t *testing.T) {
	t.Parallel()

	for _, n := range []int{17, 257, 401} {
		t.Run("complex128_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
			}

			freq := randomComplex128(n, int64(n)+7)
			dst := make([]complex128, n)

			if err := plan.Inverse(dst, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			want := reference.NaiveIDFT128(freq)
			for i := range dst {
				if diff := cmplx.Abs(dst[i] - want[i]); diff > 1e-10 {
					t.Errorf("bin %d: IFFT=%v, naive=%v, diff=%v", i, dst[i], want[i], diff)
				}
			}
		})
	}
}

// TestRader_RoundTrip covers the large sizes where the naive reference is too
// slow, including the Fermat prime 65537 (power-of-two sub-FFT) and 5-smooth
// mixed-radix sub-FFT sizes.
func TestRader_RoundTrip(t *testing.T) {
	t.Parallel()

	for _, n := range []int{3001, 4001, 12289, 40961, 65537} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
			}

			if plan.Algorithm() != algorithmRader {
				t.Fatalf("n=%d: Algorithm() = %q, want \"rader\"", n, plan.Algorithm())
			}

			src := randomComplex128(n, int64(n))
			freq := make([]complex128, n)
			back := make([]complex128, n)

			if err := plan.Forward(freq, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if err := plan.Inverse(back, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			for i := range src {
				if diff := cmplx.Abs(back[i] - src[i]); diff > 1e-10 {
					t.Fatalf("round-trip mismatch at %d: got %v, want %v (diff %v)", i, back[i], src[i], diff)
				}
			}
		})
	}
}

// TestRader_InPlace verifies dst==src operation matches out-of-place.
func TestRader_InPlace(t *testing.T) {
	t.Parallel()

	for _, n := range []int{17, 641} {
		plan, err := NewPlan[complex128](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex128(n, 99)

		want := make([]complex128, n)
		if err := plan.Forward(want, src); err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		data := make([]complex128, n)
		copy(data, src)

		if err := plan.ForwardInPlace(data); err != nil {
			t.Fatalf("ForwardInPlace failed: %v", err)
		}

		for i := range want {
			if diff := cmplx.Abs(data[i] - want[i]); diff > 1e-12 {
				t.Fatalf("n=%d: in-place mismatch at %d: got %v, want %v", n, i, data[i], want[i])
			}
		}
	}
}

// TestRader_Clone verifies cloned plans carry the Rader tables and produce
// identical results.
func TestRader_Clone(t *testing.T) {
	t.Parallel()

	n := 257

	plan, err := NewPlan[complex128](n)
	if err != nil {
		t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
	}

	exec := plan.Clone()
	defer exec.Close()

	src := randomComplex128(n, 5)

	want := make([]complex128, n)
	if err := plan.Forward(want, src); err != nil {
		t.Fatalf("plan.Forward failed: %v", err)
	}

	got := make([]complex128, n)
	if err := exec.Forward(got, src); err != nil {
		t.Fatalf("exec.Forward failed: %v", err)
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("executor mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// BenchmarkRaderVsBluestein compares the auto-selected Rader path against a
// forced Bluestein plan at the same prime sizes. The Rader eligibility gate
// (internal/fft/rader.go) was tuned from this benchmark.
func BenchmarkRaderVsBluestein(b *testing.B) {
	sizes := []int{17, 257, 641, 1601, 4001, 12289, 65537}

	run64 := func(b *testing.B, n int, opts PlanOptions) {
		b.Helper()

		plan, err := NewPlanWithOptions[complex64](n, opts)
		if err != nil {
			b.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex64(n, int64(n))
		dst := make([]complex64, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()

		for range b.N {
			_ = plan.Forward(dst, src)
		}
	}

	for _, n := range sizes {
		b.Run("Rader_"+strconv.Itoa(n), func(b *testing.B) {
			run64(b, n, PlanOptions{})
		})
		b.Run("Bluestein_"+strconv.Itoa(n), func(b *testing.B) {
			run64(b, n, PlanOptions{Strategy: KernelBluestein})
		})
	}
}

// rader7And11BenchPrimes are primes whose n-1 is 7/11-smooth but not 5-smooth,
// grouped by the power-of-two part of n-1 — the shape parameter both win gates
// (raderConvolutionWins, planner.mixedRadix7And11Wins) turn on. The list spans
// small/mid/large at each shape so the gate is fitted, not guessed.
//
//	n-1 pow2 part 2:  127, 463, 2311, 22051
//	n-1 pow2 part 4:  197, 701, 2269, 9901
//	n-1 pow2 part 8:  89, 281, 1321, 4201, 12601
//	n-1 pow2 part 16: 113, 881, 2801, 7057, 15121
//	n-1 pow2 part 32: 353, 673, 2017, 7393, 30241
//	n-1 pow2 part 64: 449, 2113, 6337, 12097
//	n-1 pow2 part >=128: 1409, 2689, 4481, 9857, 14081
//
//nolint:gochecknoglobals // benchmark input table
var rader7And11BenchPrimes = []int{
	89, 113, 127, 197, 281, 353, 449, 463, 673, 701, 881,
	1321, 1409, 2017, 2113, 2269, 2311, 2689, 2801, 4201, 4481,
	6337, 7057, 7393, 9857, 9901, 12097, 12601, 14081, 15121, 22051, 30241,
}

// BenchmarkRader7And11VsBluestein is the measurement behind the 7/11-smooth
// half of the raderConvolutionWins gate: it compares the auto-selected path
// against a forced Bluestein plan at primes whose n-1 needs a radix-7/11
// stage. Sizes the gate rejects still run here — their "Rader_" arm is then
// the Bluestein path too, so a ~1.0 ratio is the expected signature of a
// correctly gated-out shape. To re-fit the gate, widen
// raderConvolutionWins to accept every shape and re-run.
func BenchmarkRader7And11VsBluestein(b *testing.B) {
	run64 := func(b *testing.B, n int, opts PlanOptions) {
		b.Helper()

		plan, err := NewPlanWithOptions[complex64](n, opts)
		if err != nil {
			b.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex64(n, int64(n))
		dst := make([]complex64, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()

		for range b.N {
			_ = plan.Forward(dst, src)
		}
	}

	run128 := func(b *testing.B, n int, opts PlanOptions) {
		b.Helper()

		plan, err := NewPlanWithOptions[complex128](n, opts)
		if err != nil {
			b.Fatalf("NewPlan[complex128](%d) failed: %v", n, err)
		}

		src := randomComplex128(n, int64(n))
		dst := make([]complex128, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 16))
		b.ResetTimer()

		for range b.N {
			_ = plan.Forward(dst, src)
		}
	}

	for _, n := range rader7And11BenchPrimes {
		b.Run("Rader_"+strconv.Itoa(n), func(b *testing.B) {
			run64(b, n, PlanOptions{})
		})
		b.Run("Bluestein_"+strconv.Itoa(n), func(b *testing.B) {
			run64(b, n, PlanOptions{Strategy: KernelBluestein})
		})
		b.Run("Rader128_"+strconv.Itoa(n), func(b *testing.B) {
			run128(b, n, PlanOptions{})
		})
		b.Run("Bluestein128_"+strconv.Itoa(n), func(b *testing.B) {
			run128(b, n, PlanOptions{Strategy: KernelBluestein})
		})
	}
}
