//go:build amd64 && !purego

package kernels

import (
	"math"
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// radix8AVX2Sizes covers all three ladder shapes at every supported length:
// 8^k (512, 4096, 32768), 2*8^k (64, 1024, 8192) and 4*8^k (32, 256, 2048,
// 16384). Leaving one shape out would leave one whole tail stage untested.
func radix8AVX2Sizes() []int {
	return []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
}

// TestRadix8AVX2AdjacentGroupStride pins the digit-reversal identity used by
// the complex128 stage-1 gather: for an even group g, advancing to g+1 cannot
// carry out of the low unreversed digit, so the source index advances by q/8
// (or one at n=32, whose stage-1 subproblem has only one base-8 digit).
func TestRadix8AVX2AdjacentGroupStride(t *testing.T) {
	for _, n := range radix8AVX2Sizes() {
		q := n / 8
		idx := radix8GroupIndices(n)
		want := max(int32(q/8), 1)

		for g := 0; g < len(idx); g += 2 {
			if got := idx[g+1] - idx[g]; got != want {
				t.Fatalf("n=%d g=%d: adjacent stride %d, want %d", n, g, got, want)
			}
		}
	}
}

func TestRadix8AVX2ForwardMatchesReference(t *testing.T) {
	for _, n := range naiveReferenceSizes(t, radix8AVX2Sizes()) {
		src := randomComplex64(n, uint64(n))
		want := reference.NaiveDFT(src)

		dst := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, false, twiddle)

		if !forwardRadix8AVX2Complex64(dst, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: forward kernel declined", n)
		}

		tol := 2e-4 * float64(n)
		if d := maxAbsDiff64(dst, want); d > tol {
			t.Errorf("n=%d: forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix8AVX2InverseMatchesReference(t *testing.T) {
	for _, n := range naiveReferenceSizes(t, radix8AVX2Sizes()) {
		src := randomComplex64(n, uint64(n)+7)
		want := reference.NaiveIDFT(src)

		dst := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, true, twiddle)

		if !inverseRadix8AVX2Complex64(dst, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: inverse kernel declined", n)
		}

		tol := 2e-4
		if d := maxAbsDiff64(dst, want); d > tol {
			t.Errorf("n=%d: inverse max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX2LargeSizesMatchStockham cross-checks the sizes where a naive
// O(n^2) DFT is too slow to serve as the reference. Stockham is an independent
// implementation -- different algorithm, different data flow, no shared
// permutation table -- so agreement between the two is real evidence rather
// than a self-consistency check.
func TestRadix8AVX2LargeSizesMatchStockham(t *testing.T) {
	for _, n := range []int{8192, 16384, 32768} {
		src := randomComplex64(n, uint64(n)+29)

		want := make([]complex64, n)
		if !forwardStockhamComplex64(want, src, m.ComputeTwiddleFactors[complex64](n), make([]complex64, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, false, twiddle)

		if !forwardRadix8AVX2Complex64(got, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		// Both sides are O(n log n) in float32, so their disagreement grows as
		// sqrt(n) rather than as n; the same bound the radix-4 twin uses.
		tol := 4e-6 * math.Sqrt(float64(n))
		if d := maxAbsDiff64(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX2InPlace exercises the dst == src path, which routes the
// transform through scratch and copies back. The oracle is the out-of-place
// kernel: both run identical arithmetic in an identical order, so the correct
// assertion is bit-for-bit equality.
func TestRadix8AVX2InPlace(t *testing.T) {
	for _, n := range radix8AVX2Sizes() {
		src := randomComplex64(n, uint64(n)+13)

		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, false, twiddle)

		want := make([]complex64, n)
		if !forwardRadix8AVX2Complex64(want, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: out-of-place forward kernel declined", n)
		}

		buf := make([]complex64, n)
		copy(buf, src)

		if !forwardRadix8AVX2Complex64(buf, buf, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: in-place forward kernel declined", n)
		}

		for i := range buf {
			if buf[i] != want[i] {
				t.Fatalf("n=%d: in-place differs from out-of-place at %d: got %v, want %v",
					n, i, buf[i], want[i])
			}
		}
	}
}

// TestRadix8AVX2MatchesGenericLadder pins the assembly against the pure-Go
// ladder it was derived from, at every size and in both directions. The two
// share the shape classifier, the twiddle layout and the permutation table, so
// this is not an independent check of the algorithm -- the reference tests
// above are. What it does catch is the assembly disagreeing with the Go
// prototype about the arithmetic, which is the failure mode a hand-written
// butterfly actually has, and it reports it at every size rather than only the
// ones a naive DFT can afford.
func TestRadix8AVX2MatchesGenericLadder(t *testing.T) {
	for _, n := range radix8AVX2Sizes() {
		for _, inverse := range []bool{false, true} {
			src := randomComplex64(n, uint64(n)+101)

			twiddle := make([]complex64, twiddleSizeRadix8(n))
			prepareTwiddleRadix8Complex64(n, inverse, twiddle)

			want := make([]complex64, n)
			got := make([]complex64, n)

			goKernel, asmKernel := forwardRadix8Complex64, forwardRadix8AVX2Complex64
			if inverse {
				goKernel, asmKernel = inverseRadix8Complex64, inverseRadix8AVX2Complex64
			}

			if !goKernel(want, src, twiddle, make([]complex64, n)) {
				t.Fatalf("n=%d inverse=%v: generic ladder declined", n, inverse)
			}

			if !asmKernel(got, src, twiddle, make([]complex64, n)) {
				t.Fatalf("n=%d inverse=%v: avx2 kernel declined", n, inverse)
			}

			// Not bit-exact: the assembly uses FMA for the twiddle multiplies
			// and folds 1/n into stage 1, where the Go ladder folds it into the
			// same place but multiplies separately. Both are O(n log n) in
			// float32, so the gap is a random walk in sqrt(n).
			tol := 4e-6 * math.Sqrt(float64(n))
			if inverse {
				tol /= float64(n)
			}

			if d := maxAbsDiff64(got, want); d > tol {
				t.Errorf("n=%d inverse=%v: max |diff| vs generic ladder = %g, tol %g",
					n, inverse, d, tol)
			}
		}
	}
}

func TestRadix8AVX2RejectsUnsupportedSizes(t *testing.T) {
	for _, n := range []int{4, 8, 16, 12, 24, 96, 1000, 1 << 18} {
		if _, ok := radix8AVX2Limit(n); ok {
			t.Errorf("n=%d: unexpectedly reported as supported", n)
		}
	}
}
