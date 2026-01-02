//go:build arm64 && asm && !purego

package fft

import (
	"testing"
)

// benchmarkNEONSizeSpecificVsGeneric compares the performance of size-specific dispatch
// (which currently falls back to generic NEON) vs direct generic NEON calls.
// With stub implementations, these should show identical performance.
// Once unrolled kernels are implemented (phases 15.5.2-5), size-specific
// should show 5-20% speedup.
func benchmarkNEONSizeSpecificVsGeneric(b *testing.B, n int) {
	b.Run("SizeSpecific", func(b *testing.B) {
		benchmarkNEONKernel(b, n, neonSizeSpecificOrGenericDITComplex64(KernelAuto))
	})

	b.Run("GenericNEON", func(b *testing.B) {
		benchmarkNEONKernel(b, n, func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
		})
	})

	b.Run("PureGo", func(b *testing.B) {
		benchmarkNEONKernel(b, n, forwardDITComplex64)
	})
}

func benchmarkNEONKernel(b *testing.B, n int, kernel Kernel[complex64]) {
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
func BenchmarkNEONSizeSpecific_vs_Generic_16(b *testing.B) {
	benchmarkNEONSizeSpecificVsGeneric(b, 16)
}

// Benchmark size 32
func BenchmarkNEONSizeSpecific_vs_Generic_32(b *testing.B) {
	benchmarkNEONSizeSpecificVsGeneric(b, 32)
}

// Benchmark size 64
func BenchmarkNEONSizeSpecific_vs_Generic_64(b *testing.B) {
	benchmarkNEONSizeSpecificVsGeneric(b, 64)
}

// Benchmark size 128 (largest size-specific kernel)
func BenchmarkNEONSizeSpecific_vs_Generic_128(b *testing.B) {
	benchmarkNEONSizeSpecificVsGeneric(b, 128)
}

// Benchmark size 256
func BenchmarkNEONSizeSpecific_vs_Generic_256(b *testing.B) {
	benchmarkNEONSizeSpecificVsGeneric(b, 256)
}

// Benchmark size 1024 (larger size to show generic performance)
func BenchmarkNEONSizeSpecific_vs_Generic_1024(b *testing.B) {
	benchmarkNEONSizeSpecificVsGeneric(b, 1024)
}
