package math

import (
	"math/cmplx"
	"math/rand"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// randComplex64Matrix returns a random n*n complex64 slice with values well
// away from zero so a missed multiply or wrong index is very unlikely to
// pass by coincidence.
func randComplex64Matrix(rng *rand.Rand, n int) []complex64 {
	m := make([]complex64, n*n)
	for i := range m {
		m[i] = complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
	}

	return m
}

// randTwiddleTable builds a unit-circle twiddle table. It goes through
// cmplx.Exp rather than the standard math.Sin/Cos, because this package is
// itself named "math" and shadows the standard library package of the same
// name.
func randTwiddleTable(_ *rand.Rand, n int) []complex64 {
	nn := n * n
	tw := make([]complex64, nn)

	for k := range tw {
		theta := -2 * 3.141592653589793 * float64(k) / float64(nn)
		w := cmplx.Exp(complex(0, theta))
		tw[k] = complex(float32(real(w)), float32(imag(w)))
	}

	return tw
}

func referenceTransposeOOP(src []complex64, n int) []complex64 {
	dst := make([]complex64, n*n)
	for i := range n {
		for j := range n {
			dst[i*n+j] = src[j*n+i]
		}
	}

	return dst
}

func referenceTransposeTwiddle(src, twiddle []complex64, n int, conj bool) []complex64 {
	dst := make([]complex64, n*n)
	nn := n * n

	for i := range n {
		for j := range n {
			w := twiddle[(i*j)%nn]
			if conj {
				w = complex(real(w), -imag(w))
			}

			dst[i*n+j] = MulComplex64(src[j*n+i], w)
		}
	}

	return dst
}

func assertComplex64SlicesEqual(t *testing.T, got, want []complex64, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}

	for i := range want {
		d := complex128(got[i]) - complex128(want[i])
		if cmplx.Abs(d) > tol {
			t.Fatalf("index %d: got %v want %v (diff %v)", i, got[i], want[i], d)
		}
	}
}

func TestTransposeSquareOutOfPlace_PureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	for _, n := range []int{0, 1, 2, 8, 32, 64, 96, 128} {
		src := randComplex64Matrix(rng, n)
		dst := make([]complex64, n*n)

		TransposeSquareOutOfPlace(dst, src, n)

		want := referenceTransposeOOP(src, n)
		assertComplex64SlicesEqual(t, dst, want, 0)
	}
}

func TestTransposeSquareOutOfPlace_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))

	for _, n := range []int{1, 8, 32, 64, 96, 128} {
		src := randComplex64Matrix(rng, n)
		mid := make([]complex64, n*n)
		back := make([]complex64, n*n)

		TransposeSquareOutOfPlace(mid, src, n)
		TransposeSquareOutOfPlace(back, mid, n)

		assertComplex64SlicesEqual(t, back, src, 0)
	}
}

func TestTransposeSquareOutOfPlaceComplex64_Dispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(3))

	for _, n := range []int{32, 64, 96, 128} {
		src := randComplex64Matrix(rng, n)
		dst := make([]complex64, n*n)

		TransposeSquareOutOfPlaceComplex64(dst, src, n)

		want := referenceTransposeOOP(src, n)
		// Plain transpose: no arithmetic at all, exact match required.
		assertComplex64SlicesEqual(t, dst, want, 0)
	}
}

func TestTransposeSquareTwiddleComplex64_Dispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(4))

	for _, n := range []int{32, 64, 96, 128} {
		src := randComplex64Matrix(rng, n)
		twiddle := randTwiddleTable(rng, n)
		dst := make([]complex64, n*n)

		TransposeSquareTwiddleComplex64Dispatch(dst, src, twiddle, n)

		want := referenceTransposeTwiddle(src, twiddle, n, false)
		assertComplex64SlicesEqual(t, dst, want, 1e-4)
	}
}

func TestTransposeSquareTwiddleConjComplex64_Dispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(5))

	for _, n := range []int{32, 64, 96, 128} {
		src := randComplex64Matrix(rng, n)
		twiddle := randTwiddleTable(rng, n)
		dst := make([]complex64, n*n)

		TransposeSquareTwiddleConjComplex64Dispatch(dst, src, twiddle, n)

		want := referenceTransposeTwiddle(src, twiddle, n, true)
		assertComplex64SlicesEqual(t, dst, want, 1e-4)
	}
}

// TestTransposeDispatch_MatchesPureGoReference cross-checks the dispatched
// (possibly AVX2) path against the pure-Go implementation from
// transpose_oop.go directly, at exactly the sizes (64, 128) the asm claims
// to handle, plus a non-handled size (96) to prove the fallback is taken
// there too.
func TestTransposeDispatch_MatchesPureGoReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6))

	for _, n := range []int{32, 64, 96, 128} {
		src := randComplex64Matrix(rng, n)
		twiddle := randTwiddleTable(rng, n)

		gotPlain := make([]complex64, n*n)
		wantPlain := make([]complex64, n*n)
		TransposeSquareOutOfPlaceComplex64(gotPlain, src, n)
		TransposeSquareOutOfPlace(wantPlain, src, n)
		assertComplex64SlicesEqual(t, gotPlain, wantPlain, 0)

		gotTw := make([]complex64, n*n)
		wantTw := make([]complex64, n*n)
		TransposeSquareTwiddleComplex64Dispatch(gotTw, src, twiddle, n)
		TransposeSquareTwiddleComplex64(wantTw, src, twiddle, n)
		assertComplex64SlicesEqual(t, gotTw, wantTw, 1e-5)

		gotTwConj := make([]complex64, n*n)
		wantTwConj := make([]complex64, n*n)
		TransposeSquareTwiddleConjComplex64Dispatch(gotTwConj, src, twiddle, n)
		TransposeSquareTwiddleConjComplex64(wantTwConj, src, twiddle, n)
		assertComplex64SlicesEqual(t, gotTwConj, wantTwConj, 1e-5)
	}
}

func TestTransposeDispatch_ZeroAlloc(t *testing.T) {
	const n = 64

	rng := rand.New(rand.NewSource(7))
	src := randComplex64Matrix(rng, n)
	twiddle := randTwiddleTable(rng, n)
	dst := make([]complex64, n*n)

	allocs := testing.AllocsPerRun(100, func() {
		TransposeSquareOutOfPlaceComplex64(dst, src, n)
	})
	if allocs != 0 {
		t.Errorf("TransposeSquareOutOfPlaceComplex64(n=%d) allocated %v times, expected 0", n, allocs)
	}

	allocs = testing.AllocsPerRun(100, func() {
		TransposeSquareTwiddleComplex64Dispatch(dst, src, twiddle, n)
	})
	if allocs != 0 {
		t.Errorf("TransposeSquareTwiddleComplex64Dispatch(n=%d) allocated %v times, expected 0", n, allocs)
	}

	allocs = testing.AllocsPerRun(100, func() {
		TransposeSquareTwiddleConjComplex64Dispatch(dst, src, twiddle, n)
	})
	if allocs != 0 {
		t.Errorf("TransposeSquareTwiddleConjComplex64Dispatch(n=%d) allocated %v times, expected 0", n, allocs)
	}
}

// TestTransposeDispatch_AVX2AvailableOnThisMachine reports whether the tests
// above exercised the asm at all, which now takes two conditions rather than
// one: the AVX2 dispatch is compiled in only under `-tags fftprobe` (see
// transpose_amd64.go), and the CPU has to support AVX2. Without both, every
// dispatch test above is a test of the pure-Go fallback — worth stating,
// because those tests pass either way.
func TestTransposeDispatch_AVX2AvailableOnThisMachine(t *testing.T) {
	features := cpu.DetectFeatures()
	t.Logf("AVX2 transpose dispatch linked (-tags fftprobe): %v; AVX2 available on this machine: %v",
		transposeAVX2Linked, features.HasAVX2)

	if transposeAVX2Linked && !features.HasAVX2 {
		t.Log("probe build on a non-AVX2 host: the asm was not executed")
	}
}
