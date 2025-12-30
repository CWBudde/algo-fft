package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	radix4Tol64  = 1e-4
	radix4Tol128 = 1e-10
)

func TestRadix4ForwardMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{4, 16, 64} {
		src := randomComplex64(n, 0xC0FFEE+uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardRadix4Complex64(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix4Complex64 failed for n=%d", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64Close(t, dst, want, radix4Tol64)
	}
}

func TestRadix4InverseMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{4, 16, 64} {
		src := randomComplex64(n, 0xFEEDBEEF+uint64(n))
		fwd := make([]complex64, n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardRadix4Complex64(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix4Complex64 failed for n=%d", n)
		}

		if !inverseRadix4Complex64(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseRadix4Complex64 failed for n=%d", n)
		}

		want := reference.NaiveIDFT(fwd)
		assertComplex64Close(t, dst, want, radix4Tol64)
	}
}

func TestRadix4ForwardMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{4, 16, 64} {
		src := randomComplex128(n, 0xBEEFBEEF+uint64(n))
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardRadix4Complex128(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix4Complex128 failed for n=%d", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128Close(t, dst, want, radix4Tol128)
	}
}

func TestRadix4InverseMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{4, 16, 64} {
		src := randomComplex128(n, 0xDEADBEEF+uint64(n))
		fwd := make([]complex128, n)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardRadix4Complex128(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix4Complex128 failed for n=%d", n)
		}

		if !inverseRadix4Complex128(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseRadix4Complex128 failed for n=%d", n)
		}

		want := reference.NaiveIDFT128(fwd)
		assertComplex128Close(t, dst, want, radix4Tol128)
	}
}

func BenchmarkRadix4Forward_1024(b *testing.B) {
	benchmarkForwardKernel(b, 1024, radix4Forward[complex64])
}

func BenchmarkRadix2Forward_1024(b *testing.B) {
	benchmarkForwardKernel(b, 1024, ditForward[complex64])
}

func BenchmarkRadix4Forward_4096(b *testing.B) {
	benchmarkForwardKernel(b, 4096, radix4Forward[complex64])
}

func BenchmarkRadix2Forward_4096(b *testing.B) {
	benchmarkForwardKernel(b, 4096, ditForward[complex64])
}

func benchmarkForwardKernel(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
	src := randomComplex64(n, 0x1234+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !IsPowerOf2(n) {
		b.Fatalf("benchmark expects power-of-two length, got %d", n)
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
