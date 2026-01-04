//go:build arm64 && asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func BenchmarkNEONForward_64(b *testing.B)      { benchmarkNEONForward(b, 64) }
func BenchmarkNEONForward_256(b *testing.B)     { benchmarkNEONForward(b, 256) }
func BenchmarkNEONForward_1024(b *testing.B)    { benchmarkNEONForward(b, 1024) }
func BenchmarkNEONForward_4096(b *testing.B)    { benchmarkNEONForward(b, 4096) }
func BenchmarkGoForward_64(b *testing.B)        { benchmarkGoForward(b, 64) }
func BenchmarkGoForward_256(b *testing.B)       { benchmarkGoForward(b, 256) }
func BenchmarkGoForward_1024(b *testing.B)      { benchmarkGoForward(b, 1024) }
func BenchmarkGoForward_4096(b *testing.B)      { benchmarkGoForward(b, 4096) }
func BenchmarkNEONInverse_64(b *testing.B)      { benchmarkNEONInverse(b, 64) }
func BenchmarkNEONInverse_256(b *testing.B)     { benchmarkNEONInverse(b, 256) }
func BenchmarkNEONInverse_1024(b *testing.B)    { benchmarkNEONInverse(b, 1024) }
func BenchmarkNEONInverse_4096(b *testing.B)    { benchmarkNEONInverse(b, 4096) }
func BenchmarkGoInverse_64(b *testing.B)        { benchmarkGoInverse(b, 64) }
func BenchmarkGoInverse_256(b *testing.B)       { benchmarkGoInverse(b, 256) }
func BenchmarkGoInverse_1024(b *testing.B)      { benchmarkGoInverse(b, 1024) }
func BenchmarkGoInverse_4096(b *testing.B)      { benchmarkGoInverse(b, 4096) }
func BenchmarkNEONForward128_64(b *testing.B)   { benchmarkNEONForward128(b, 64) }
func BenchmarkNEONForward128_256(b *testing.B)  { benchmarkNEONForward128(b, 256) }
func BenchmarkNEONForward128_1024(b *testing.B) { benchmarkNEONForward128(b, 1024) }
func BenchmarkNEONForward128_4096(b *testing.B) { benchmarkNEONForward128(b, 4096) }
func BenchmarkGoForward128_64(b *testing.B)     { benchmarkGoForward128(b, 64) }
func BenchmarkGoForward128_256(b *testing.B)    { benchmarkGoForward128(b, 256) }
func BenchmarkGoForward128_1024(b *testing.B)   { benchmarkGoForward128(b, 1024) }
func BenchmarkGoForward128_4096(b *testing.B)   { benchmarkGoForward128(b, 4096) }
func BenchmarkNEONInverse128_64(b *testing.B)   { benchmarkNEONInverse128(b, 64) }
func BenchmarkNEONInverse128_256(b *testing.B)  { benchmarkNEONInverse128(b, 256) }
func BenchmarkNEONInverse128_1024(b *testing.B) { benchmarkNEONInverse128(b, 1024) }
func BenchmarkNEONInverse128_4096(b *testing.B) { benchmarkNEONInverse128(b, 4096) }
func BenchmarkGoInverse128_64(b *testing.B)     { benchmarkGoInverse128(b, 64) }
func BenchmarkGoInverse128_256(b *testing.B)    { benchmarkGoInverse128(b, 256) }
func BenchmarkGoInverse128_1024(b *testing.B)   { benchmarkGoInverse128(b, 1024) }
func BenchmarkGoInverse128_4096(b *testing.B)   { benchmarkGoInverse128(b, 4096) }

func benchmarkNEONForward(b *testing.B, n int) {
	benchmarkKernelForward(b, n, cpu.Features{
		HasNEON:      true,
		Architecture: "arm64",
	})
}

func benchmarkGoForward(b *testing.B, n int) {
	benchmarkDITForward(b, n)
}

func benchmarkNEONInverse(b *testing.B, n int) {
	benchmarkKernelInverse(b, n, cpu.Features{
		HasNEON:      true,
		Architecture: "arm64",
	})
}

func benchmarkGoInverse(b *testing.B, n int) {
	benchmarkDITInverse(b, n)
}

func benchmarkNEONForward128(b *testing.B, n int) {
	benchmarkKernelForward128(b, n, true)
}

func benchmarkGoForward128(b *testing.B, n int) {
	benchmarkKernelForward128(b, n, false)
}

func benchmarkNEONInverse128(b *testing.B, n int) {
	benchmarkKernelInverse128(b, n, true)
}

func benchmarkGoInverse128(b *testing.B, n int) {
	benchmarkKernelInverse128(b, n, false)
}

func benchmarkKernelForward(b *testing.B, n int, features cpu.Features) {
	b.Helper()
	b.Cleanup(cpu.ResetDetection)
	cpu.SetForcedFeatures(features)

	kernels := SelectKernels[complex64](cpu.DetectFeatures())
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}
	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for b.Loop() {
		if !kernels.Forward(dst, src, twiddle, scratch, bitrev) {
			b.Fatalf("Forward kernel returned false for n=%d", n)
		}
	}
}

func benchmarkKernelInverse(b *testing.B, n int, features cpu.Features) {
	b.Helper()
	b.Cleanup(cpu.ResetDetection)
	cpu.SetForcedFeatures(features)

	kernels := SelectKernels[complex64](cpu.DetectFeatures())
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}
	freq := make([]complex64, n)

	if !kernels.Forward(freq, src, twiddle, scratch, bitrev) {
		b.Fatalf("Forward kernel returned false for n=%d", n)
	}

	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for b.Loop() {
		if !kernels.Inverse(dst, freq, twiddle, scratch, bitrev) {
			b.Fatalf("Inverse kernel returned false for n=%d", n)
		}
	}
}

func benchmarkDITForward(b *testing.B, n int) {
	b.Helper()

	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}
	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for b.Loop() {
		if !forwardDITComplex64(dst, src, twiddle, scratch, bitrev) {
			b.Fatalf("forwardDITComplex64 returned false for n=%d", n)
		}
	}
}

func benchmarkDITInverse(b *testing.B, n int) {
	b.Helper()

	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}
	freq := make([]complex64, n)

	if !forwardDITComplex64(freq, src, twiddle, scratch, bitrev) {
		b.Fatalf("forwardDITComplex64 returned false for n=%d", n)
	}

	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for b.Loop() {
		if !inverseDITComplex64(dst, freq, twiddle, scratch, bitrev) {
			b.Fatalf("inverseDITComplex64 returned false for n=%d", n)
		}
	}
}

func benchmarkKernelForward128(b *testing.B, n int, useNEON bool) {
	b.Helper()

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}
	dst := make([]complex128, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for b.Loop() {
		if useNEON {
			if !forwardNEONComplex128Asm(dst, src, twiddle, scratch, bitrev) {
				b.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}
		} else {
			if !forwardDITComplex128(dst, src, twiddle, scratch, bitrev) {
				b.Fatalf("forwardDITComplex128 returned false for n=%d", n)
			}
		}
	}
}

func benchmarkKernelInverse128(b *testing.B, n int, useNEON bool) {
	b.Helper()

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}
	freq := make([]complex128, n)

	if !forwardDITComplex128(freq, src, twiddle, scratch, bitrev) {
		b.Fatalf("forwardDITComplex128 returned false for n=%d", n)
	}

	dst := make([]complex128, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for b.Loop() {
		if useNEON {
			if !inverseNEONComplex128Asm(dst, freq, twiddle, scratch, bitrev) {
				b.Fatalf("inverseNEONComplex128Asm returned false for n=%d", n)
			}
		} else {
			if !inverseDITComplex128(dst, freq, twiddle, scratch, bitrev) {
				b.Fatalf("inverseDITComplex128 returned false for n=%d", n)
			}
		}
	}
}
