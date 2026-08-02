//go:build fftprobe

package math

import (
	"math/rand"
	"strconv"
	"testing"
)

// The comparison benchmark that goes with the probe-gated AVX2 transposes.
// §2.2 asks a probe-gated kernel to carry both a correctness test and a
// benchmark against what it would replace; the correctness half already lives
// in transpose_oop_test.go (it runs in either configuration and exercises the
// asm in this one), so this file supplies the other half.
//
// What it measures: the dispatched path (AVX2 asm at n = 64 and 128) against
// the pure-Go implementation it would displace, for all three variants —
// plain, fused twiddle, fused conjugate twiddle. n = 96 is included as a
// control: the dispatch does not handle it, so both columns run the same code
// and any gap there is measurement noise rather than a speedup.
//
//	go test -tags fftprobe -run '^$' -bench BenchmarkTransposeProbe ./internal/math/
//
// Phase 3 is what decides these kernels: if six-step/four-step wires them in,
// the tag comes off and this file goes with it.
func benchTransposeSizes() []int { return []int{64, 96, 128} }

func BenchmarkTransposeProbePlain(b *testing.B) {
	for _, n := range benchTransposeSizes() {
		rng := rand.New(rand.NewSource(11))
		src := randComplex64Matrix(rng, n)
		dst := make([]complex64, n*n)

		b.Run(benchName("dispatch", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				TransposeSquareOutOfPlaceComplex64(dst, src, n)
			}
		})

		b.Run(benchName("purego", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				TransposeSquareOutOfPlace(dst, src, n)
			}
		})
	}
}

func BenchmarkTransposeProbeTwiddle(b *testing.B) {
	for _, n := range benchTransposeSizes() {
		rng := rand.New(rand.NewSource(12))
		src := randComplex64Matrix(rng, n)
		twiddle := randTwiddleTable(rng, n)
		dst := make([]complex64, n*n)

		b.Run(benchName("dispatch", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				TransposeSquareTwiddleComplex64Dispatch(dst, src, twiddle, n)
			}
		})

		b.Run(benchName("purego", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				TransposeSquareTwiddleComplex64(dst, src, twiddle, n)
			}
		})
	}
}

func BenchmarkTransposeProbeTwiddleConj(b *testing.B) {
	for _, n := range benchTransposeSizes() {
		rng := rand.New(rand.NewSource(13))
		src := randComplex64Matrix(rng, n)
		twiddle := randTwiddleTable(rng, n)
		dst := make([]complex64, n*n)

		b.Run(benchName("dispatch", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				TransposeSquareTwiddleConjComplex64Dispatch(dst, src, twiddle, n)
			}
		})

		b.Run(benchName("purego", n), func(b *testing.B) {
			b.SetBytes(int64(n * n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				TransposeSquareTwiddleConjComplex64(dst, src, twiddle, n)
			}
		})
	}
}

// benchName keeps the two columns adjacent under benchstat.
func benchName(impl string, n int) string {
	return impl + "/n=" + strconv.Itoa(n)
}
