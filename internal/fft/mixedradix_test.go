package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestMixedRadixForwardMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{6, 10, 12, 15, 20, 30, 60} {
		src := randomComplex64(n, 0xBADC0DE+uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := make([]int, n)

		if !forwardMixedRadixComplex64(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardMixedRadixComplex64 failed for n=%d", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64SliceClose(t, dst, want, n)
	}
}

func TestMixedRadixInverseMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{6, 10, 12, 15, 20, 30, 60} {
		src := randomComplex64(n, 0xC0FFEE+uint64(n))
		fwd := make([]complex64, n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := make([]int, n)

		if !forwardMixedRadixComplex64(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardMixedRadixComplex64 failed for n=%d", n)
		}

		if !inverseMixedRadixComplex64(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseMixedRadixComplex64 failed for n=%d", n)
		}

		want := reference.NaiveIDFT(fwd)
		assertComplex64SliceClose(t, dst, want, n)
	}
}

func TestMixedRadixForwardMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{6, 10, 12, 15, 20, 30, 60} {
		src := randomComplex128(n, 0xC001D00D+uint64(n))
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := make([]int, n)

		if !forwardMixedRadixComplex128(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardMixedRadixComplex128 failed for n=%d", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128SliceClose(t, dst, want, n)
	}
}

func TestMixedRadixInverseMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{6, 10, 12, 15, 20, 30, 60} {
		src := randomComplex128(n, 0xF00DBAAD+uint64(n))
		fwd := make([]complex128, n)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := make([]int, n)

		if !forwardMixedRadixComplex128(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardMixedRadixComplex128 failed for n=%d", n)
		}

		if !inverseMixedRadixComplex128(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseMixedRadixComplex128 failed for n=%d", n)
		}

		want := reference.NaiveIDFT128(fwd)
		assertComplex128SliceClose(t, dst, want, n)
	}
}

func BenchmarkMixedRadixForward_60(b *testing.B) {
	benchmarkMixedRadixKernel(b, 60, mixedRadixForward[complex64])
}

func BenchmarkMixedRadixForward_60_Padded64(b *testing.B) {
	benchmarkMixedRadixPaddedKernel(b, 60, 64, ditForward[complex64])
}

func benchmarkMixedRadixKernel(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
	src := randomComplex64(n, 0x1234+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := make([]int, n)

	if !isHighlyComposite(n) {
		b.Fatalf("benchmark expects mixed-radix length, got %d", n)
	}

	b.SetBytes(int64(n * 8))
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		if !kernel(dst, src, twiddle, scratch, bitrev) {
			b.Fatalf("kernel failed for n=%d", n)
		}
	}
}

func benchmarkMixedRadixPaddedKernel(b *testing.B, n, padded int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
	if padded < n {
		b.Fatalf("padded length %d must be >= %d", padded, n)
	}

	src := randomComplex64(padded, 0x1234+uint64(n))
	for i := n; i < padded; i++ {
		src[i] = 0
	}

	dst := make([]complex64, padded)
	scratch := make([]complex64, padded)
	twiddle := ComputeTwiddleFactors[complex64](padded)
	bitrev := ComputeBitReversalIndices(padded)

	if !IsPowerOf2(padded) {
		b.Fatalf("padded length must be power of two, got %d", padded)
	}

	b.SetBytes(int64(n * 8))
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		if !kernel(dst, src, twiddle, scratch, bitrev) {
			b.Fatalf("kernel failed for n=%d", padded)
		}
	}
}
