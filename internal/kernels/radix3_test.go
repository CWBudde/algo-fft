package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestRadix3ForwardMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{3, 9, 27} {
		src := randomComplex64(n, 0xCAFE+uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := make([]int, n)

		if !forwardRadix3Complex64(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix3Complex64 failed for n=%d", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestRadix3InverseMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{3, 9, 27} {
		src := randomComplex64(n, 0xBEEF+uint64(n))
		fwd := make([]complex64, n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := make([]int, n)

		if !forwardRadix3Complex64(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix3Complex64 failed for n=%d", n)
		}

		if !inverseRadix3Complex64(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseRadix3Complex64 failed for n=%d", n)
		}

		want := reference.NaiveIDFT(fwd)
		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestRadix3ForwardMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{3, 9, 27} {
		src := randomComplex128(n, 0xDEAD+uint64(n))
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := make([]int, n)

		if !forwardRadix3Complex128(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix3Complex128 failed for n=%d", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128Close(t, dst, want, 1e-10)
	}
}

func TestRadix3InverseMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{3, 9, 27} {
		src := randomComplex128(n, 0xF00D+uint64(n))
		fwd := make([]complex128, n)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := make([]int, n)

		if !forwardRadix3Complex128(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix3Complex128 failed for n=%d", n)
		}

		if !inverseRadix3Complex128(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseRadix3Complex128 failed for n=%d", n)
		}

		want := reference.NaiveIDFT128(fwd)
		assertComplex128Close(t, dst, want, 1e-10)
	}
}

func BenchmarkRadix3Forward_27(b *testing.B) {
	benchmarkRadix3ForwardKernel(b, 27, radix3Forward[complex64])
}

func BenchmarkRadix3Forward_243(b *testing.B) {
	benchmarkRadix3ForwardKernel(b, 243, radix3Forward[complex64])
}

func benchmarkRadix3ForwardKernel(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
	b.Helper()

	src := randomComplex64(n, 0x1234+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := make([]int, n)

	if !isPowerOf3(n) {
		b.Fatalf("benchmark expects power-of-three length, got %d", n)
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
