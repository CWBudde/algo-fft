package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestRadix4Complex64 validates the pure-Go Radix-4 kernel.
func TestRadix4Complex64(t *testing.T) {
	t.Parallel()

	// Sizes must be powers of 4: 4, 16, 64, 256, 1024, 4096
	sizes := []int{4, 16, 64, 256, 1024}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xDEADBEEF+uint64(n))
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !forwardRadix4Complex64(dst, src, twiddle, scratch) {
				t.Fatalf("forwardRadix4Complex64 failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, 1e-4)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xCAFEBABE+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !forwardRadix4Complex64(fwd, src, twiddle, scratch) {
				t.Fatalf("forwardRadix4Complex64 failed for n=%d", n)
			}

			if !inverseRadix4Complex64(dst, fwd, twiddle, scratch) {
				t.Fatalf("inverseRadix4Complex64 failed for n=%d", n)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64Close(t, dst, want, 1e-4)
		})
	}
}

// TestRadix4Complex128 validates the pure-Go Radix-4 kernel (double precision).
func TestRadix4Complex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 16, 64, 256, 1024}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0xBEEFCAFE+uint64(n))
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !forwardRadix4Complex128(dst, src, twiddle, scratch) {
				t.Fatalf("forwardRadix4Complex128 failed for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128Close(t, dst, want, 1e-10)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0xFEEDFACE+uint64(n))
			fwd := make([]complex128, n)
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !forwardRadix4Complex128(fwd, src, twiddle, scratch) {
				t.Fatalf("forwardRadix4Complex128 failed for n=%d", n)
			}

			if !inverseRadix4Complex128(dst, fwd, twiddle, scratch) {
				t.Fatalf("inverseRadix4Complex128 failed for n=%d", n)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128Close(t, dst, want, 1e-10)
		})
	}
}

func BenchmarkRadix4(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(testName("radix4", n), func(b *testing.B) {
			benchmarkForwardKernel(b, n, radix4Forward[complex64])
		})
		b.Run(testName("generic", n), func(b *testing.B) {
			benchmarkForwardKernel(b, n, ditForward[complex64])
		})
	}
}

func benchmarkForwardKernel(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex64) bool) {
	b.Helper()

	src := randomComplex64(n, 0)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	b.ResetTimer()

	for range b.N {
		kernel(dst, src, twiddle, scratch)
	}
}
