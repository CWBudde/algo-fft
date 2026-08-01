//go:build amd64 && !purego && fftprobe

package fft

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// radix4C128ProbeSizes spans both shapes the pair supports at sizes worth
// measuring: powers of four (64, 256, 1024, 4096) take the pure radix-4
// kernel, the rest take the radix-4-then-2 mixed kernel.
var radix4C128ProbeSizes = []int{64, 128, 256, 512, 1024, 2048, 4096, 8192}

func radix4C128ProbeInput(n int) []complex128 {
	src := make([]complex128, n)

	rng := rand.New(rand.NewPCG(uint64(uint(n)), 1))
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	return src
}

// TestRadix4C128ProbeAccepts is the guard that matters most here: the kernels
// are unreachable from production, so nothing else in the suite would notice
// them silently declining. A probe that always returns false would benchmark
// nothing and report no error.
func TestRadix4C128ProbeAccepts(t *testing.T) {
	t.Parallel()

	if _, _, ok := getAVX2Kernels128(); !ok {
		t.Skip("AVX2 not available")
	}

	for _, n := range radix4C128ProbeSizes {
		src := radix4C128ProbeInput(n)
		twiddle, scratch := prepareFFTData[complex128](n)
		dst := make([]complex128, n)

		if !forwardAVX2Complex128Radix4Probe(dst, src, twiddle, scratch) {
			t.Errorf("n=%d: forward radix-4 declined", n)
		}

		if !inverseAVX2Complex128Radix4Probe(dst, src, twiddle, scratch) {
			t.Errorf("n=%d: inverse radix-4 declined", n)
		}
	}
}

// closeEnough compares against the naive DFT with a tolerance relative to the
// reference magnitude.
//
// getToleranceForSize128 is an absolute bound, which is right for the tests
// that compare two FFTs against each other -- they accumulate error the same
// way, so their difference stays small in absolute terms. Against the naive
// DFT it is the wrong shape: the two algorithms sum n terms in a different
// order, and with random input on [-1,1] a length-8192 spectrum reaches
// magnitude ~150, so an honest 1e-12 relative agreement shows up as a 5e-10
// absolute difference and fails a 5e-10 absolute bound.
func closeEnough(got, want complex128, tol float64) bool {
	scale := cmplx.Abs(want)
	if scale < 1 {
		scale = 1
	}

	return cmplx.Abs(got-want) <= tol*scale
}

// TestRadix4C128ProbeMatchesReference validates each direction against the
// naive DFT independently, rather than relying on a round-trip: a matched pair
// of sign errors round-trips perfectly. See docs/TESTING.md.
func TestRadix4C128ProbeMatchesReference(t *testing.T) {
	t.Parallel()

	if _, _, ok := getAVX2Kernels128(); !ok {
		t.Skip("AVX2 not available")
	}

	for _, n := range radix4C128ProbeSizes {
		src := radix4C128ProbeInput(n)
		twiddle, scratch := prepareFFTData[complex128](n)
		tol := getToleranceForSize128(n)

		fwd := make([]complex128, n)
		if !forwardAVX2Complex128Radix4Probe(fwd, src, twiddle, scratch) {
			t.Fatalf("n=%d: forward declined", n)
		}

		want := reference.NaiveDFT128(src)
		for i := range want {
			if !closeEnough(fwd[i], want[i], tol) {
				t.Errorf("n=%d forward[%d]: got %v want %v (tol %v)", n, i, fwd[i], want[i], tol)

				break
			}
		}

		inv := make([]complex128, n)
		if !inverseAVX2Complex128Radix4Probe(inv, src, twiddle, scratch) {
			t.Fatalf("n=%d: inverse declined", n)
		}

		wantInv := reference.NaiveIDFT128(src)
		for i := range wantInv {
			if !closeEnough(inv[i], wantInv[i], tol) {
				t.Errorf("n=%d inverse[%d]: got %v want %v (tol %v)", n, i, inv[i], wantInv[i], tol)

				break
			}
		}
	}
}

func benchRadix4C128(b *testing.B, n int, fn func(dst, src, twiddle, scratch []complex128) bool) {
	b.Helper()

	src := radix4C128ProbeInput(n)
	twiddle, scratch := prepareFFTData[complex128](n)
	dst := make([]complex128, n)

	if !fn(dst, src, twiddle, scratch) {
		b.Skipf("n=%d: kernel declined", n)
	}

	b.SetBytes(int64(n * 16))
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		fn(dst, src, twiddle, scratch)
	}
}

// The four benchmarks below run in one process so the ratio is a same-process
// comparison. Read radix-4 over radix-2 per size and direction; do not compare
// absolute figures across hosts.
func BenchmarkC128Radix4Forward(b *testing.B) {
	for _, n := range radix4C128ProbeSizes {
		b.Run(sizeString(n), func(b *testing.B) {
			benchRadix4C128(b, n, forwardAVX2Complex128Radix4Probe)
		})
	}
}

func BenchmarkC128Radix2Forward(b *testing.B) {
	for _, n := range radix4C128ProbeSizes {
		b.Run(sizeString(n), func(b *testing.B) {
			benchRadix4C128(b, n, forwardAVX2Complex128Radix2Probe)
		})
	}
}

func BenchmarkC128Radix4Inverse(b *testing.B) {
	for _, n := range radix4C128ProbeSizes {
		b.Run(sizeString(n), func(b *testing.B) {
			benchRadix4C128(b, n, inverseAVX2Complex128Radix4Probe)
		})
	}
}

func BenchmarkC128Radix2Inverse(b *testing.B) {
	for _, n := range radix4C128ProbeSizes {
		b.Run(sizeString(n), func(b *testing.B) {
			benchRadix4C128(b, n, inverseAVX2Complex128Radix2Probe)
		})
	}
}
