package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	//nolint:unused
	radix5Tol64 = 1e-4
	//nolint:unused
	radix5Tol128 = 1e-10
)

func TestRadix5ForwardMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{5, 25, 125} {
		src := randomComplex64(n, 0xC001+uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := make([]int, n)

		if !forwardRadix5Complex64(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix5Complex64 failed for n=%d", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestRadix5InverseMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{5, 25, 125} {
		src := randomComplex64(n, 0xB00B+uint64(n))
		fwd := make([]complex64, n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := make([]int, n)

		if !forwardRadix5Complex64(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix5Complex64 failed for n=%d", n)
		}

		if !inverseRadix5Complex64(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseRadix5Complex64 failed for n=%d", n)
		}

		want := reference.NaiveIDFT(fwd)
		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestRadix5ForwardMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{5, 25, 125} {
		src := randomComplex128(n, 0xD00D+uint64(n))
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := make([]int, n)

		if !forwardRadix5Complex128(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix5Complex128 failed for n=%d", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128Close(t, dst, want, 1e-10)
	}
}

func TestRadix5InverseMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{5, 25, 125} {
		src := randomComplex128(n, 0xFACE+uint64(n))
		fwd := make([]complex128, n)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := make([]int, n)

		if !forwardRadix5Complex128(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardRadix5Complex128 failed for n=%d", n)
		}

		if !inverseRadix5Complex128(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseRadix5Complex128 failed for n=%d", n)
		}

		want := reference.NaiveIDFT128(fwd)
		assertComplex128Close(t, dst, want, 1e-10)
	}
}

func BenchmarkRadix5Forward_125(b *testing.B) {
	benchmarkRadix5ForwardKernel(b, 125, radix5Forward[complex64])
}

func BenchmarkRadix5Forward_625(b *testing.B) {
	benchmarkRadix5ForwardKernel(b, 625, radix5Forward[complex64])
}

func benchmarkRadix5ForwardKernel(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
	src := randomComplex64(n, 0x1234+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := make([]int, n)

	if !isPowerOf5(n) {
		b.Fatalf("benchmark expects power-of-five length, got %d", n)
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
