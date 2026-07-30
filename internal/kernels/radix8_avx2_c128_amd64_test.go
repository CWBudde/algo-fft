//go:build amd64 && !purego

package kernels

import (
	"math"
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestRadix8AVX2Complex128ForwardMatchesReference(t *testing.T) {
	for _, n := range naiveReferenceSizes(t, radix8AVX2Sizes()) {
		src := randomComplex128(n, uint64(n))
		want := reference.NaiveDFT128(src)

		dst := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, false, twiddle)

		if !forwardRadix8AVX2Complex128(dst, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: forward kernel declined", n)
		}

		tol := 1e-10 * float64(n)
		if d := maxAbsDiff128(dst, want); d > tol {
			t.Errorf("n=%d: forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix8AVX2Complex128InverseMatchesReference(t *testing.T) {
	for _, n := range naiveReferenceSizes(t, radix8AVX2Sizes()) {
		src := randomComplex128(n, uint64(n)+7)
		want := reference.NaiveIDFT128(src)

		dst := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, true, twiddle)

		if !inverseRadix8AVX2Complex128(dst, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: inverse kernel declined", n)
		}

		tol := 1e-10
		if d := maxAbsDiff128(dst, want); d > tol {
			t.Errorf("n=%d: inverse max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX2Complex128LargeSizesMatchStockham is the complex128 twin of
// TestRadix8AVX2LargeSizesMatchStockham: an independent algorithm at the sizes
// where a naive O(n^2) DFT is too slow to serve as the reference.
func TestRadix8AVX2Complex128LargeSizesMatchStockham(t *testing.T) {
	for _, n := range []int{8192, 16384, 32768} {
		src := randomComplex128(n, uint64(n)+29)

		want := make([]complex128, n)
		if !forwardStockhamComplex128(want, src, m.ComputeTwiddleFactors[complex128](n), make([]complex128, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex128, n)
		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, false, twiddle)

		if !forwardRadix8AVX2Complex128(got, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		tol := 1e-14 * math.Sqrt(float64(n))
		if d := maxAbsDiff128(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix8AVX2Complex128InPlace exercises the dst == src path.
func TestRadix8AVX2Complex128InPlace(t *testing.T) {
	for _, n := range radix8AVX2Sizes() {
		src := randomComplex128(n, uint64(n)+13)

		twiddle := make([]complex128, twiddleSizeRadix8(n))
		prepareTwiddleRadix8Complex128(n, false, twiddle)

		want := make([]complex128, n)
		if !forwardRadix8AVX2Complex128(want, src, twiddle, make([]complex128, n)) {
			t.Fatalf("n=%d: out-of-place forward kernel declined", n)
		}

		buf := make([]complex128, n)
		copy(buf, src)

		if !forwardRadix8AVX2Complex128(buf, buf, twiddle, make([]complex128, n)) {
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

// TestRadix8AVX2Complex128MatchesGenericLadder pins the assembly against the
// pure-Go ladder at every size and in both directions; see the complex64 twin
// for why this sits alongside the reference tests rather than replacing them.
func TestRadix8AVX2Complex128MatchesGenericLadder(t *testing.T) {
	for _, n := range radix8AVX2Sizes() {
		for _, inverse := range []bool{false, true} {
			src := randomComplex128(n, uint64(n)+101)

			twiddle := make([]complex128, twiddleSizeRadix8(n))
			prepareTwiddleRadix8Complex128(n, inverse, twiddle)

			want := make([]complex128, n)
			got := make([]complex128, n)

			goKernel, asmKernel := forwardRadix8Complex128, forwardRadix8AVX2Complex128
			if inverse {
				goKernel, asmKernel = inverseRadix8Complex128, inverseRadix8AVX2Complex128
			}

			if !goKernel(want, src, twiddle, make([]complex128, n)) {
				t.Fatalf("n=%d inverse=%v: generic ladder declined", n, inverse)
			}

			if !asmKernel(got, src, twiddle, make([]complex128, n)) {
				t.Fatalf("n=%d inverse=%v: avx2 kernel declined", n, inverse)
			}

			tol := 1e-14 * math.Sqrt(float64(n))
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
