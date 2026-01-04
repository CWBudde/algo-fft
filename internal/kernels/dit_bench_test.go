package kernels

import (
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"testing"
)

// benchCase64 defines a benchmark case for complex64 kernels.
type benchCase64 struct {
	name    string
	n       int
	bitrev  func(int) []int
	forward func(dst, src, twiddle, scratch []complex64, bitrev []int) bool
	inverse func(dst, src, twiddle, scratch []complex64, bitrev []int) bool
}

// benchCase128 defines a benchmark case for complex128 kernels.
type benchCase128 struct {
	name    string
	n       int
	bitrev  func(int) []int
	forward func(dst, src, twiddle, scratch []complex128, bitrev []int) bool
	inverse func(dst, src, twiddle, scratch []complex128, bitrev []int) bool
}

// BenchmarkDITComplex64 benchmarks all Go DIT kernels for complex64.
func BenchmarkDITComplex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT4Radix4Complex64, inverseDIT4Radix4Complex64},
		{"Size8/Radix2", 8, mathpkg.ComputeBitReversalIndices, forwardDIT8Radix2Complex64, inverseDIT8Radix2Complex64},
		{"Size8/Radix4", 8, mathpkg.ComputeBitReversalIndicesMixed24, forwardDIT8Radix4Complex64, inverseDIT8Radix4Complex64},
		{"Size16/Radix2", 16, mathpkg.ComputeBitReversalIndices, forwardDIT16Complex64, inverseDIT16Complex64},
		{"Size16/Radix4", 16, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT16Radix4Complex64, inverseDIT16Radix4Complex64},
		{"Size32/Radix2", 32, mathpkg.ComputeBitReversalIndices, forwardDIT32Complex64, inverseDIT32Complex64},
		{"Size32/Radix4x4x2", 32, mathpkg.ComputeBitReversalIndicesMixed24, forwardDIT32MixedRadix24Complex64, inverseDIT32MixedRadix24Complex64},
		{"Size64/Radix2", 64, mathpkg.ComputeBitReversalIndices, forwardDIT64Complex64, inverseDIT64Complex64},
		{"Size64/Radix4", 64, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT64Radix4Complex64, inverseDIT64Radix4Complex64},
		{"Size128", 128, mathpkg.ComputeBitReversalIndices, forwardDIT128Complex64, inverseDIT128Complex64},
		{"Size256/Radix2", 256, mathpkg.ComputeBitReversalIndices, forwardDIT256Complex64, inverseDIT256Complex64},
		{"Size256/Radix4", 256, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT256Radix4Complex64, inverseDIT256Radix4Complex64},
		{"Size512", 512, mathpkg.ComputeBitReversalIndices, forwardDIT512Complex64, inverseDIT512Complex64},
	}

	for _, tc := range cases {
		b.Run(tc.name+"/Forward", func(b *testing.B) {
			runBenchComplex64(b, tc.n, tc.bitrev, tc.forward)
		})
		b.Run(tc.name+"/Inverse", func(b *testing.B) {
			runBenchComplex64(b, tc.n, tc.bitrev, tc.inverse)
		})
	}
}

// BenchmarkDITComplex128 benchmarks all Go DIT kernels for complex128.
func BenchmarkDITComplex128(b *testing.B) {
	cases := []benchCase128{
		{"Size4/Radix4", 4, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT4Radix4Complex128, inverseDIT4Radix4Complex128},
		{"Size8/Radix2", 8, mathpkg.ComputeBitReversalIndices, forwardDIT8Radix2Complex128, inverseDIT8Radix2Complex128},
		{"Size8/Radix4", 8, mathpkg.ComputeBitReversalIndicesMixed24, forwardDIT8Radix4Complex128, inverseDIT8Radix4Complex128},
		{"Size16/Radix2", 16, mathpkg.ComputeBitReversalIndices, forwardDIT16Complex128, inverseDIT16Complex128},
		{"Size16/Radix4", 16, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT16Radix4Complex128, inverseDIT16Radix4Complex128},
		{"Size32/Radix2", 32, mathpkg.ComputeBitReversalIndices, forwardDIT32Complex128, inverseDIT32Complex128},
		{"Size32/Radix4x4x2", 32, mathpkg.ComputeBitReversalIndicesMixed24, forwardDIT32MixedRadix24Complex128, inverseDIT32MixedRadix24Complex128},
		{"Size64/Radix2", 64, mathpkg.ComputeBitReversalIndices, forwardDIT64Complex128, inverseDIT64Complex128},
		{"Size64/Radix4", 64, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT64Radix4Complex128, inverseDIT64Radix4Complex128},
		{"Size128", 128, mathpkg.ComputeBitReversalIndices, forwardDIT128Complex128, inverseDIT128Complex128},
		{"Size256/Radix2", 256, mathpkg.ComputeBitReversalIndices, forwardDIT256Complex128, inverseDIT256Complex128},
		{"Size256/Radix4", 256, mathpkg.ComputeBitReversalIndicesRadix4, forwardDIT256Radix4Complex128, inverseDIT256Radix4Complex128},
		{"Size512", 512, mathpkg.ComputeBitReversalIndices, forwardDIT512Complex128, inverseDIT512Complex128},
	}

	for _, tc := range cases {
		b.Run(tc.name+"/Forward", func(b *testing.B) {
			runBenchComplex128(b, tc.n, tc.bitrev, tc.forward)
		})
		b.Run(tc.name+"/Inverse", func(b *testing.B) {
			runBenchComplex128(b, tc.n, tc.bitrev, tc.inverse)
		})
	}
}

func runBenchComplex64(b *testing.B, n int, bitrev func(int) []int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	br := bitrev(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for b.Loop() {
		kernel(dst, src, twiddle, scratch, br)
	}
}

func runBenchComplex128(b *testing.B, n int, bitrev func(int) []int, kernel func(dst, src, twiddle, scratch []complex128, bitrev []int) bool) {
	src := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	br := bitrev(n)

	for i := range src {
		src[i] = complex(float64(i), float64(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 16))

	for b.Loop() {
		kernel(dst, src, twiddle, scratch, br)
	}
}
