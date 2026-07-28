//go:build amd64 && !purego

package kernels

import (
	"math/cmplx"
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func maxAbsDiff128(got, want []complex128) float64 {
	worst := 0.0

	for i := range got {
		if d := cmplx.Abs(got[i] - want[i]); d > worst {
			worst = d
		}
	}

	return worst
}

func TestRadix4AVX2Complex128ForwardMatchesReference(t *testing.T) {
	for _, n := range radix4AVX2Sizes() {
		src := randomComplex128(n, uint64(n))
		want := reference.NaiveDFT128(src)

		dst := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
		scratch := make([]complex128, n)
		prepareTwiddleRadix4AVX2Complex128(n, false, twiddle)

		if !forwardRadix4AVX2Complex128(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: forward kernel declined", n)
		}

		// float64 accumulation over n terms: the naive reference is itself the
		// less accurate of the two, so the tolerance tracks its error growth.
		tol := 1e-12 * float64(n)
		if d := maxAbsDiff128(dst, want); d > tol {
			t.Errorf("n=%d: forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix4AVX2Complex128InverseMatchesReference(t *testing.T) {
	for _, n := range radix4AVX2Sizes() {
		src := randomComplex128(n, uint64(n)+7)
		want := reference.NaiveIDFT128(src)

		dst := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
		scratch := make([]complex128, n)
		prepareTwiddleRadix4AVX2Complex128(n, true, twiddle)

		if !inverseRadix4AVX2Complex128(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: inverse kernel declined", n)
		}

		tol := 1e-12
		if d := maxAbsDiff128(dst, want); d > tol {
			t.Errorf("n=%d: inverse max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix4AVX2Complex128InPlace exercises the dst == src path, which routes
// the transform through scratch and copies back.
func TestRadix4AVX2Complex128InPlace(t *testing.T) {
	for _, n := range radix4AVX2Sizes() {
		src := randomComplex128(n, uint64(n)+13)
		want := reference.NaiveDFT128(src)

		buf := make([]complex128, n)
		copy(buf, src)

		twiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
		scratch := make([]complex128, n)
		prepareTwiddleRadix4AVX2Complex128(n, false, twiddle)

		if !forwardRadix4AVX2Complex128(buf, buf, twiddle, scratch) {
			t.Fatalf("n=%d: in-place forward kernel declined", n)
		}

		tol := 1e-12 * float64(n)
		if d := maxAbsDiff128(buf, want); d > tol {
			t.Errorf("n=%d: in-place forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix4AVX2Complex128MatchesStockham cross-checks the sizes where a naive
// O(n^2) DFT is too slow to serve as the reference. Stockham is an independent
// implementation -- different algorithm, different data flow, no shared
// permutation table -- so agreement between the two is real evidence rather
// than a self-consistency check.
func TestRadix4AVX2Complex128MatchesStockham(t *testing.T) {
	for _, n := range []int{8192, 16384, 32768, 65536} {
		src := randomComplex128(n, uint64(n)+29)

		want := make([]complex128, n)
		if !forwardStockhamComplex128(want, src, m.ComputeTwiddleFactors[complex128](n), make([]complex128, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
		prepareTwiddleRadix4AVX2Complex128(n, false, twiddle)

		if !forwardRadix4AVX2Complex128(got, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		tol := 1e-9 * float64(n)
		if d := maxAbsDiff128(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix4AVX2Complex128RoundTrip65536 covers the largest supported size end
// to end.
func TestRadix4AVX2Complex128RoundTrip65536(t *testing.T) {
	const n = 65536

	src := randomComplex128(n, 31)

	fwdTwiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
	prepareTwiddleRadix4AVX2Complex128(n, false, fwdTwiddle)

	invTwiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
	prepareTwiddleRadix4AVX2Complex128(n, true, invTwiddle)

	spectrum := make([]complex128, n)
	if !forwardRadix4AVX2Complex128(spectrum, src, fwdTwiddle, make([]complex128, n)) {
		t.Fatal("forward declined")
	}

	back := make([]complex128, n)
	if !inverseRadix4AVX2Complex128(back, spectrum, invTwiddle, make([]complex128, n)) {
		t.Fatal("inverse declined")
	}

	if d := maxAbsDiff128(back, src); d > 1e-11 {
		t.Errorf("round-trip max |diff| = %g", d)
	}
}
