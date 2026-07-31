//go:build amd64 && !purego

package kernels

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// requireAVX512 skips a test on a host without AVX-512. The development laptop
// is Alder Lake and has none, so every assertion below is dead there and the
// kernel's only real gate is the Xeon; saying so out loud is the point of the
// explicit skip message.
func requireAVX512(t *testing.T) {
	t.Helper()

	if !cpu.DetectFeatures().HasAVX512 {
		t.Skip("host has no AVX-512")
	}
}

// radix8AVX512Sizes64 covers all three ladder shapes at every length the
// complex64 kernel accepts: 8^k (512, 4096, 32768), 2*8^k (64, 1024, 8192) and
// 4*8^k (256, 2048, 16384). Leaving one shape out would leave one whole tail
// stage untested.
func radix8AVX512Sizes64() []int {
	return []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
}

// radix8AVX512Sizes128 adds n = 32, which the complex128 kernel reaches and the
// complex64 one does not.
func radix8AVX512Sizes128() []int {
	return append([]int{32}, radix8AVX512Sizes64()...)
}

// naiveReferenceCap bounds the sizes handed to the O(n²) reference DFT.
//
// The reference sweeps below are the only tests in this file that cost more
// than milliseconds, and their cost is dominated entirely by their largest
// entry: over 64..32768 the term n² sums to 1.42e9, of which n = 32768 alone is
// 1.07e9 and everything up to 4096 is 1.5%. Four such sweeps (two directions x
// two precisions) added roughly 200 s to a package that already sat near Go's
// 10-minute default timeout, and pushed it past on a two-core host.
//
// Capping here costs no coverage, because it is the division of labour the
// large-size tests below already assume: 8192, 16384 and 32768 are cross-checked
// against Stockham, an independent implementation with a different data flow and
// no shared permutation table, which is stronger evidence at those sizes than a
// slower copy of the same assertion. What the naive sweep uniquely provides is
// ground truth that shares nothing with any FFT in this tree, and 64/128/256
// already exercise all three ladder shapes -- 8^k, 2*8^k and 4*8^k -- with their
// distinct tail stages.
const naiveReferenceCap = 4096

// radix8AVX512NaiveSizes filters a size list down to what the O(n²) reference
// can afford.
func radix8AVX512NaiveSizes(t *testing.T, sizes []int) []int {
	t.Helper()

	kept := make([]int, 0, len(sizes))

	for _, n := range sizes {
		if n <= naiveReferenceCap {
			kept = append(kept, n)
		}
	}

	return naiveReferenceSizes(t, kept)
}

func TestRadix8AVX512ForwardMatchesReference(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512NaiveSizes(t, radix8AVX512Sizes64()) {
		src := randomComplex64(n, uint64(n))
		want := reference.NaiveDFT(src)

		dst := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, false, twiddle)

		if !forwardRadix8AVX512Complex64(dst, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: forward kernel declined", n)
		}

		tol := 2e-4 * float64(n)
		if d := maxAbsDiff64(dst, want); d > tol {
			t.Errorf("n=%d: forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix8AVX512InverseMatchesReference(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512NaiveSizes(t, radix8AVX512Sizes64()) {
		src := randomComplex64(n, uint64(n)+7)
		want := reference.NaiveIDFT(src)

		dst := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, true, twiddle)

		if !inverseRadix8AVX512Complex64(dst, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: inverse kernel declined", n)
		}

		tol := 2e-4
		if d := maxAbsDiff64(dst, want); d > tol {
			t.Errorf("n=%d: inverse max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix8AVX512Complex128ForwardMatchesReference(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512NaiveSizes(t, radix8AVX512Sizes128()) {
		src := randomComplex128(n, uint64(n)+3)
		want := reference.NaiveDFT128(src)

		dst := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, false, twiddle)

		if !forwardRadix8AVX512Complex128(dst, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: forward kernel declined", n)
		}

		tol := 1e-12 * float64(n)
		if d := maxAbsDiff128(dst, want); d > tol {
			t.Errorf("n=%d: forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix8AVX512Complex128InverseMatchesReference(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512NaiveSizes(t, radix8AVX512Sizes128()) {
		src := randomComplex128(n, uint64(n)+11)
		want := reference.NaiveIDFT128(src)

		dst := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, true, twiddle)

		if !inverseRadix8AVX512Complex128(dst, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: inverse kernel declined", n)
		}

		tol := 1e-12
		if d := maxAbsDiff128(dst, want); d > tol {
			t.Errorf("n=%d: inverse max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX512LargeSizesMatchStockham cross-checks the sizes where a naive
// O(n^2) DFT is too slow to serve as the reference. Stockham is an independent
// implementation -- different algorithm, different data flow, no shared
// permutation table -- so agreement between the two is real evidence rather
// than a self-consistency check.
func TestRadix8AVX512LargeSizesMatchStockham(t *testing.T) {
	requireAVX512(t)

	for _, n := range []int{8192, 16384, 32768} {
		src := randomComplex64(n, uint64(n)+29)

		want := make([]complex64, n)
		if !forwardStockhamComplex64(want, src, m.ComputeTwiddleFactors[complex64](n), make([]complex64, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, false, twiddle)

		if !forwardRadix8AVX512Complex64(got, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		// Both sides are O(n log n) in float32, so their disagreement grows as
		// sqrt(n) rather than as n; the same bound the AVX2 twin uses.
		tol := 4e-6 * math.Sqrt(float64(n))
		if d := maxAbsDiff64(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX512Complex128LargeSizesMatchStockham is the complex128 twin,
// and carries the large-size ground truth that naiveReferenceCap removes from
// the reference sweep above.
func TestRadix8AVX512Complex128LargeSizesMatchStockham(t *testing.T) {
	requireAVX512(t)

	for _, n := range []int{8192, 16384, 32768} {
		src := randomComplex128(n, uint64(n)+31)

		want := make([]complex128, n)
		if !forwardStockhamComplex128(want, src, m.ComputeTwiddleFactors[complex128](n), make([]complex128, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, false, twiddle)

		if !forwardRadix8AVX512Complex128(got, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		tol := 1e-13 * math.Sqrt(float64(n))
		if d := maxAbsDiff128(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX512InPlace exercises the dst == src path, which routes the
// transform through scratch and copies back. The oracle is the out-of-place
// kernel: both run identical arithmetic in an identical order, so the correct
// assertion is bit-for-bit equality.
func TestRadix8AVX512InPlace(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512Sizes64() {
		src := randomComplex64(n, uint64(n)+13)

		twiddle := make([]complex64, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex64(n, false, twiddle)

		want := make([]complex64, n)
		if !forwardRadix8AVX512Complex64(want, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: out-of-place forward kernel declined", n)
		}

		buf := make([]complex64, n)
		copy(buf, src)

		if !forwardRadix8AVX512Complex64(buf, buf, twiddle, make([]complex64, n)) {
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

// TestRadix8AVX512Complex128InPlace is the complex128 twin of the above.
func TestRadix8AVX512Complex128InPlace(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512Sizes128() {
		src := randomComplex128(n, uint64(n)+17)

		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, false, twiddle)

		want := make([]complex128, n)
		if !forwardRadix8AVX512Complex128(want, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: out-of-place forward kernel declined", n)
		}

		buf := make([]complex128, n)
		copy(buf, src)

		if !forwardRadix8AVX512Complex128(buf, buf, twiddle, make([]complex128, n)) {
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

// TestRadix8AVX512MatchesGenericLadder pins the assembly against the pure-Go
// ladder it was derived from, at every size and in both directions. The two
// share the shape classifier, the twiddle layout and the permutation table, so
// this is not an independent check of the algorithm -- the reference tests
// above are. What it does catch is the assembly disagreeing with the Go
// prototype about the arithmetic, which is the failure mode a hand-written
// butterfly actually has, and it reports it at every size rather than only the
// ones a naive DFT can afford.
func TestRadix8AVX512MatchesGenericLadder(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512Sizes64() {
		for _, inverse := range []bool{false, true} {
			src := randomComplex64(n, uint64(n)+101)

			twiddle := make([]complex64, twiddleSizeRadix8(n))
			prepareTwiddleRadix8Complex64(n, inverse, twiddle)

			want := make([]complex64, n)
			got := make([]complex64, n)

			goKernel, asmKernel := forwardRadix8Complex64, forwardRadix8AVX512Complex64
			if inverse {
				goKernel, asmKernel = inverseRadix8Complex64, inverseRadix8AVX512Complex64
			}

			if !goKernel(want, src, twiddle, make([]complex64, n)) {
				t.Fatalf("n=%d inverse=%v: generic ladder declined", n, inverse)
			}

			if !asmKernel(got, src, twiddle, make([]complex64, n)) {
				t.Fatalf("n=%d inverse=%v: avx512 kernel declined", n, inverse)
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

// TestRadix8AVX512Complex128MatchesGenericLadder is the complex128 twin.
func TestRadix8AVX512Complex128MatchesGenericLadder(t *testing.T) {
	requireAVX512(t)

	for _, n := range radix8AVX512Sizes128() {
		for _, inverse := range []bool{false, true} {
			src := randomComplex128(n, uint64(n)+103)

			twiddle := make([]complex128, twiddleSizeRadix8(n))
			prepareTwiddleRadix8Complex128(n, inverse, twiddle)

			want := make([]complex128, n)
			got := make([]complex128, n)

			goKernel, asmKernel := forwardRadix8Complex128, forwardRadix8AVX512Complex128
			if inverse {
				goKernel, asmKernel = inverseRadix8Complex128, inverseRadix8AVX512Complex128
			}

			if !goKernel(want, src, twiddle, make([]complex128, n)) {
				t.Fatalf("n=%d inverse=%v: generic ladder declined", n, inverse)
			}

			if !asmKernel(got, src, twiddle, make([]complex128, n)) {
				t.Fatalf("n=%d inverse=%v: avx512 kernel declined", n, inverse)
			}

			tol := 1e-13 * math.Sqrt(float64(n))
			if inverse {
				tol /= float64(n)
			}

			if d := maxAbsDiff128(got, want); d > tol {
				t.Errorf("n=%d inverse=%v: max |diff| vs generic ladder = %g, tol %g",
					n, inverse, d, tol)
			}
		}
	}
}

func TestRadix8AVX512RejectsUnsupportedSizes(t *testing.T) {
	for _, n := range []int{4, 8, 16, 32, 12, 24, 96, 1000, 1 << 18} {
		if _, ok := radix8AVX512Limit64(n); ok {
			t.Errorf("complex64 n=%d: unexpectedly reported as supported", n)
		}
	}

	for _, n := range []int{4, 8, 16, 12, 24, 96, 1000, 1 << 18} {
		if _, ok := radix8AVX512Limit128(n); ok {
			t.Errorf("complex128 n=%d: unexpectedly reported as supported", n)
		}
	}
}
