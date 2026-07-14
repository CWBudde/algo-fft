//go:build amd64 && !purego

// AVX-512 generic kernel benchmarks, with same-shape AVX2 comparison runs:
//   - Radix2 (generic radix-2, apples-to-apples with the AVX-512 kernel)
//   - Auto (the AVX2 generic entry point, which prefers radix-4/mixed)
//
// Tests are in asm_amd64_avx512_test.go.
package fft

import (
	"fmt"
	"testing"
)

//nolint:gochecknoglobals // shared read-only benchmark fixture
var avx512BenchSizes = []int{64, 256, 1024, 4096, 16384}

func benchKernelComplex64(b *testing.B, n int, kernel Kernel[complex64]) {
	b.Helper()

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		if !kernel(dst, src, twiddle, scratch) {
			b.Fatal("kernel returned false")
		}
	}
}

func benchKernelComplex128(b *testing.B, n int, kernel Kernel[complex128]) {
	b.Helper()

	src := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	for i := range src {
		src[i] = complex(float64(i), float64(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 16))

	for range b.N {
		if !kernel(dst, src, twiddle, scratch) {
			b.Fatal("kernel returned false")
		}
	}
}

func BenchmarkAVX512GenericComplex64(b *testing.B) {
	requireAVX512(b)

	for _, n := range avx512BenchSizes {
		b.Run(fmt.Sprintf("Size%d/AVX512", n), func(b *testing.B) {
			benchKernelComplex64(b, n, forwardAVX512Complex64)
		})
		b.Run(fmt.Sprintf("Size%d/AVX2Radix2", n), func(b *testing.B) {
			benchKernelComplex64(b, n, forwardAVX2Complex64GenericRadix2Asm)
		})
		b.Run(fmt.Sprintf("Size%d/AVX2Auto", n), func(b *testing.B) {
			benchKernelComplex64(b, n, forwardAVX2Complex64Asm)
		})
	}
}

func BenchmarkAVX512GenericComplex128(b *testing.B) {
	requireAVX512(b)

	for _, n := range avx512BenchSizes {
		b.Run(fmt.Sprintf("Size%d/AVX512", n), func(b *testing.B) {
			benchKernelComplex128(b, n, forwardAVX512Complex128)
		})
		b.Run(fmt.Sprintf("Size%d/AVX2Radix2", n), func(b *testing.B) {
			benchKernelComplex128(b, n, forwardAVX2Complex128Asm)
		})
	}
}

func BenchmarkAVX512InverseComplex64(b *testing.B) {
	requireAVX512(b)

	for _, n := range avx512BenchSizes {
		b.Run(fmt.Sprintf("Size%d/AVX512", n), func(b *testing.B) {
			benchKernelComplex64(b, n, inverseAVX512Complex64)
		})
		b.Run(fmt.Sprintf("Size%d/AVX2Auto", n), func(b *testing.B) {
			benchKernelComplex64(b, n, inverseAVX2Complex64Asm)
		})
	}
}
