//go:build amd64 && asm && !purego

package fft

import (
	"testing"
)

// benchmarkSizeSpecificVsGeneric compares the performance of size-specific dispatch
// (which currently falls back to generic AVX2) vs direct generic AVX2 calls.
// With stub implementations, these should show identical performance.
// Once unrolled kernels are implemented (phases 14.5.2-14.5.5), size-specific
// should show 5-20% speedup.
func benchmarkSizeSpecificVsGeneric(b *testing.B, n int) {
	b.Run("SizeSpecific", func(b *testing.B) {
		benchmarkKernel(b, n, avx2SizeSpecificOrGenericDITComplex64(KernelAuto))
	})

	b.Run("GenericAVX2", func(b *testing.B) {
		benchmarkKernel(b, n, func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		})
	})

	b.Run("PureGo", func(b *testing.B) {
		benchmarkKernel(b, n, forwardDITComplex64)
	})
}

func benchmarkKernel(b *testing.B, n int, kernel Kernel[complex64]) {
	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndices(n)

	// Initialize with random data
	for i := range src {
		src[i] = complex(float32(i), float32(n-i))
	}

	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // 8 bytes per complex64

	b.ResetTimer()
	for range b.N {
		if !kernel(dst, src, twiddle, scratch, bitrev) {
			b.Fatal("kernel returned false")
		}
	}
}

// Benchmark size 16 (smallest size-specific kernel)
func BenchmarkAVX2SizeSpecific_vs_Generic_16(b *testing.B) {
	benchmarkSizeSpecificVsGeneric(b, 16)
}

// Benchmark size 32
func BenchmarkAVX2SizeSpecific_vs_Generic_32(b *testing.B) {
	benchmarkSizeSpecificVsGeneric(b, 32)
}

// Benchmark size 64
func BenchmarkAVX2SizeSpecific_vs_Generic_64(b *testing.B) {
	benchmarkSizeSpecificVsGeneric(b, 64)
}

// Benchmark size 128 (largest size-specific kernel)
func BenchmarkAVX2SizeSpecific_vs_Generic_128(b *testing.B) {
	benchmarkSizeSpecificVsGeneric(b, 128)
}

// Benchmark size 256 (should use generic AVX2, not size-specific)
func BenchmarkAVX2SizeSpecific_vs_Generic_256(b *testing.B) {
	benchmarkSizeSpecificVsGeneric(b, 256)
}

// Benchmark size 1024 (larger size to show generic performance)
func BenchmarkAVX2SizeSpecific_vs_Generic_1024(b *testing.B) {
	benchmarkSizeSpecificVsGeneric(b, 1024)
}
