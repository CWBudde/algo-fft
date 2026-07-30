//go:build amd64 && !purego

package kernels

import (
	"math"
	"math/cmplx"
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func radix4AVX2Sizes() []int {
	return []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
}

func maxAbsDiff64(got, want []complex64) float64 {
	worst := 0.0

	for i := range got {
		if d := cmplx.Abs(complex128(got[i]) - complex128(want[i])); d > worst {
			worst = d
		}
	}

	return worst
}

// TestRadix4AVX2GroupIndices checks the identity the kernel relies on: the full
// radix-4 digit-reversal permutation p satisfies p[4g+d] = p[4g] + d*(n/4), so
// only every fourth entry has to be tabulated.
func TestRadix4AVX2GroupIndices(t *testing.T) {
	for _, n := range radix4AVX2Sizes() {
		limit, ok := radix4AVX2Limit(n)
		if !ok {
			t.Fatalf("n=%d: unexpectedly unsupported", n)
		}

		// A power of four uses the plain radix-4 digit reversal; 2*4^k uses the
		// radix-4-then-2 one. Picking the wrong table here would make the test
		// pass against a permutation the kernel never uses.
		full := m.ComputeBitReversalIndicesRadix4(n)
		if limit != n {
			full = m.ComputeBitReversalIndicesRadix4Then2(n)
		}

		if full == nil {
			t.Fatalf("n=%d: no permutation available", n)
		}

		idx := radix4GroupIndices(n)
		if len(idx) != n/4 {
			t.Fatalf("n=%d: got %d group indices, want %d", n, len(idx), n/4)
		}

		quarter := n / 4

		for g := range quarter {
			for d := range 4 {
				want := full[4*g+d]
				if got := int(idx[g]) + d*quarter; got != want {
					t.Fatalf("n=%d g=%d d=%d: got %d, want %d", n, g, d, got, want)
				}
			}
		}
	}
}

// TestRadix4AVX2MatchesStockham cross-checks the sizes whose naive O(n²) DFT
// is skipped under the race detector (and, at 65536, is skipped everywhere).
// Stockham is an independent implementation -- different algorithm, different
// data flow, no shared permutation table -- so agreement between the two is
// real evidence rather than a self-consistency check. This is the complex64
// twin of TestRadix4AVX2Complex128MatchesStockham, which predates it; without
// it the complex64 kernel had no large-size cross-check at all.
func TestRadix4AVX2MatchesStockham(t *testing.T) {
	for _, n := range []int{8192, 16384, 32768, 65536} {
		src := randomComplex64(n, uint64(n)+29)

		want := make([]complex64, n)
		if !forwardStockhamComplex64(want, src, m.ComputeTwiddleFactors[complex64](n), make([]complex64, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
		prepareTwiddleRadix4AVX2(n, false, twiddle)

		if !forwardRadix4AVX2Complex64(got, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		// Both sides are O(n log n) in float32, so their disagreement grows as
		// sqrt(n) (random-walk rounding), not as n. Measured agreement is
		// 3.8e-5 at n=8192 rising to 1.4e-4 at n=65536, which this bound
		// clears by ~8x. The 2e-4*n convention used against the *naive* DFT
		// would be ~5e4x looser than the observed error here and would accept
		// a wrong bin as rounding -- it is calibrated for a reference that is
		// itself the inaccurate side, which Stockham is not.
		tol := 4e-6 * math.Sqrt(float64(n))
		if d := maxAbsDiff64(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix4AVX2ForwardMatchesReference(t *testing.T) {
	for _, n := range naiveReferenceSizes(t, radix4AVX2Sizes()) {
		src := randomComplex64(n, uint64(n))
		want := reference.NaiveDFT(src)

		dst := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
		scratch := make([]complex64, n)
		prepareTwiddleRadix4AVX2(n, false, twiddle)

		if !forwardRadix4AVX2Complex64(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: forward kernel declined", n)
		}

		// float32 accumulation over n terms; tolerance scales with sqrt(n).
		tol := 2e-4 * float64(n)
		if d := maxAbsDiff64(dst, want); d > tol {
			t.Errorf("n=%d: forward max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

func TestRadix4AVX2InverseMatchesReference(t *testing.T) {
	for _, n := range naiveReferenceSizes(t, radix4AVX2Sizes()) {
		src := randomComplex64(n, uint64(n)+7)
		want := reference.NaiveIDFT(src)

		dst := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
		scratch := make([]complex64, n)
		prepareTwiddleRadix4AVX2(n, true, twiddle)

		if !inverseRadix4AVX2Complex64(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: inverse kernel declined", n)
		}

		tol := 2e-4
		if d := maxAbsDiff64(dst, want); d > tol {
			t.Errorf("n=%d: inverse max |diff| = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix4AVX2InPlace exercises the dst == src path, which routes the
// transform through scratch and copies back.
//
// The oracle is the out-of-place kernel, not the naive DFT: what is under test
// here is that aliasing dst to src changes nothing, and the two paths run
// identical arithmetic in an identical order, so the correct assertion is
// bit-for-bit equality. Comparing against the naive DFT instead would both
// cost O(n²) -- enough to dominate the race-detector run at these sizes -- and
// be *weaker*, since a tolerance wide enough for float32 accumulation would
// wave through a real aliasing defect as rounding. Kernel correctness proper
// is TestRadix4AVX2ForwardMatchesReference's job.
func TestRadix4AVX2InPlace(t *testing.T) {
	for _, n := range radix4AVX2Sizes() {
		src := randomComplex64(n, uint64(n)+13)

		twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
		prepareTwiddleRadix4AVX2(n, false, twiddle)

		want := make([]complex64, n)
		if !forwardRadix4AVX2Complex64(want, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: out-of-place forward kernel declined", n)
		}

		buf := make([]complex64, n)
		copy(buf, src)

		if !forwardRadix4AVX2Complex64(buf, buf, twiddle, make([]complex64, n)) {
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

func TestRadix4AVX2RejectsUnsupportedSizes(t *testing.T) {
	for _, n := range []int{4, 8, 12, 24, 96, 1000, 1 << 18} {
		if radix4AVX2SizeOK(n) {
			t.Errorf("n=%d: unexpectedly reported as supported", n)
		}
	}
}

// TestRadix4AVX2LargeSizesMatchStockham cross-checks the sizes where a naive
// O(n^2) DFT is too slow to serve as the reference. Stockham is an independent
// implementation -- different algorithm, different data flow, no shared
// permutation table -- so agreement between the two is real evidence rather
// than a self-consistency check.
func TestRadix4AVX2LargeSizesMatchStockham(t *testing.T) {
	for _, n := range []int{8192, 16384, 32768, 65536} {
		src := randomComplex64(n, uint64(n)+29)

		want := make([]complex64, n)
		if !forwardStockhamComplex64(want, src, m.ComputeTwiddleFactors[complex64](n), make([]complex64, 2*n)) {
			t.Fatalf("n=%d: stockham reference declined", n)
		}

		got := make([]complex64, n)
		twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
		prepareTwiddleRadix4AVX2(n, false, twiddle)

		if !forwardRadix4AVX2Complex64(got, src, twiddle, make([]complex64, n)) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		// Both accumulate in float32 over log2(n) stages; the gap between two
		// correct implementations grows with n but stays far below the signal.
		tol := 2e-3 * float64(n)
		if d := maxAbsDiff64(got, want); d > tol {
			t.Errorf("n=%d: max |diff| vs stockham = %g, tol %g", n, d, tol)
		}
	}
}

// TestRadix4AVX2RoundTrip65536 covers the largest supported size end to end.
func TestRadix4AVX2RoundTrip65536(t *testing.T) {
	const n = 65536

	src := randomComplex64(n, 31)

	fwdTwiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
	prepareTwiddleRadix4AVX2(n, false, fwdTwiddle)

	invTwiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
	prepareTwiddleRadix4AVX2(n, true, invTwiddle)

	spectrum := make([]complex64, n)
	if !forwardRadix4AVX2Complex64(spectrum, src, fwdTwiddle, make([]complex64, n)) {
		t.Fatal("forward declined")
	}

	back := make([]complex64, n)
	if !inverseRadix4AVX2Complex64(back, spectrum, invTwiddle, make([]complex64, n)) {
		t.Fatal("inverse declined")
	}

	if d := maxAbsDiff64(back, src); d > 1e-3 {
		t.Errorf("round-trip max |diff| = %g", d)
	}
}

// radix4AVX2Reference adapts the 256-bit radix-4 kernel to the plain
// (dst, src, twiddle, scratch) shape by preparing its own twiddle table. Tests
// that used the removed per-size radix-4 codelets as an independent cross-check
// call this instead; the twiddle argument is ignored.
func radix4AVX2Reference(dst, src, _, scratch []complex64) bool {
	n := len(src)

	twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
	prepareTwiddleRadix4AVX2(n, false, twiddle)

	return forwardRadix4AVX2Complex64(dst, src, twiddle, scratch)
}
