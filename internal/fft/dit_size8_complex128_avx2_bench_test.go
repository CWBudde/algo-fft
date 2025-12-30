//go:build amd64 && fft_asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

var benchSink128 complex128

func benchmarkAVX2Size8Complex128(b *testing.B, fn func(dst, src, twiddle, scratch []complex128, bitrev []int) bool) {
	if !cpu.DetectFeatures().HasAVX2 {
		b.Skip("AVX2 not available on this machine")
	}

	const n = 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1)*0.75, float64(i+1)*0.33)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	b.ReportAllocs()
	b.SetBytes(n * 16) // bytes processed in src (complex128)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if !fn(dst, src, twiddle, scratch, bitrev) {
			b.Fatal("kernel returned false")
		}
		// Prevent overly aggressive optimization
		benchSink128 += dst[i&(n-1)]
	}
}

func BenchmarkAVX2Size8Radix2Complex128_Forward(b *testing.B) {
	benchmarkAVX2Size8Complex128(b, forwardAVX2Size8Radix2Complex128Asm)
}

func BenchmarkAVX2Size8Radix8Complex128_Forward(b *testing.B) {
	benchmarkAVX2Size8Complex128(b, forwardAVX2Size8Radix8Complex128Asm)
}

func benchmarkAVX2Size8Complex128Inverse(b *testing.B, fn func(dst, src, twiddle, scratch []complex128, bitrev []int) bool) {
	if !cpu.DetectFeatures().HasAVX2 {
		b.Skip("AVX2 not available on this machine")
	}

	const n = 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1)*0.75, float64(i+1)*0.33)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)
	freq := make([]complex128, n)

	// Precompute a valid frequency-domain input (so inverse is exercised meaningfully)
	if !forwardAVX2Size8Radix8Complex128Asm(freq, src, twiddle, scratch, bitrev) {
		b.Fatal("setup forward failed")
	}

	b.ReportAllocs()
	b.SetBytes(n * 16)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if !fn(dst, freq, twiddle, scratch, bitrev) {
			b.Fatal("kernel returned false")
		}
		benchSink128 += dst[i&(n-1)]
	}
}

func BenchmarkAVX2Size8Radix2Complex128_Inverse(b *testing.B) {
	benchmarkAVX2Size8Complex128Inverse(b, inverseAVX2Size8Radix2Complex128Asm)
}

func BenchmarkAVX2Size8Radix8Complex128_Inverse(b *testing.B) {
	benchmarkAVX2Size8Complex128Inverse(b, inverseAVX2Size8Radix8Complex128Asm)
}

func BenchmarkGoSize8Radix8Complex128_Forward(b *testing.B) {
	const n = 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1)*0.75, float64(i+1)*0.33)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	b.ReportAllocs()
	b.SetBytes(n * 16)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if !forwardDIT8Radix8Complex128(dst, src, twiddle, scratch, bitrev) {
			b.Fatal("kernel returned false")
		}
		benchSink128 += dst[i&(n-1)]
	}
}

func BenchmarkGoSize8Radix8Complex128_Inverse(b *testing.B) {
	const n = 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1)*0.75, float64(i+1)*0.33)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)
	freq := make([]complex128, n)

	if !forwardDIT8Radix8Complex128(freq, src, twiddle, scratch, bitrev) {
		b.Fatal("setup forward failed")
	}

	b.ReportAllocs()
	b.SetBytes(n * 16)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if !inverseDIT8Radix8Complex128(dst, freq, twiddle, scratch, bitrev) {
			b.Fatal("kernel returned false")
		}
		benchSink128 += dst[i&(n-1)]
	}
}
