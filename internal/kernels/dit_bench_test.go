package kernels

import (
	"testing"
)

// benchCase64 defines a benchmark case for complex64 kernels.
type benchCase64 struct {
	name    string
	n       int
	forward func(dst, src, twiddle, scratch []complex64) bool
	inverse func(dst, src, twiddle, scratch []complex64) bool
}

// benchCase128 defines a benchmark case for complex128 kernels.
type benchCase128 struct {
	name    string
	n       int
	forward func(dst, src, twiddle, scratch []complex128) bool
	inverse func(dst, src, twiddle, scratch []complex128) bool
}

// BenchmarkDITComplex64 benchmarks all Go DIT kernels for complex64.
func BenchmarkDITComplex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, forwardDIT4Radix4Complex64, inverseDIT4Radix4Complex64},
		{"Size8/Radix2", 8, forwardDIT8Radix2Complex64, inverseDIT8Radix2Complex64},
		{"Size8/Radix4", 8, forwardDIT8Radix4Complex64, inverseDIT8Radix4Complex64},
		{"Size8/Radix8", 8, forwardDIT8Radix8Complex64, inverseDIT8Radix8Complex64},
		{"Size16/Radix2", 16, forwardDIT16Radix2Complex64, inverseDIT16Radix2Complex64},
		{"Size16/Radix4", 16, forwardDIT16Radix4Complex64, inverseDIT16Radix4Complex64},
		{"Size32/Radix2", 32, forwardDIT32Complex64, inverseDIT32Complex64},
		{"Size32/Radix4x4x2", 32, forwardDIT32Radix4Then2Complex64, inverseDIT32Radix4Then2Complex64},
		{"Size64/Radix2", 64, forwardDIT64Complex64, inverseDIT64Complex64},
		{"Size64/Radix4", 64, forwardDIT64Radix4Complex64, inverseDIT64Radix4Complex64},
		{"Size128", 128, forwardDIT128Complex64, inverseDIT128Complex64},
		{"Size256/Radix2", 256, forwardDIT256Complex64, inverseDIT256Complex64},
		{"Size256/Radix4", 256, forwardDIT256Radix4Complex64, inverseDIT256Radix4Complex64},
		{"Size256/Radix16", 256, forwardDIT256Radix16Complex64, inverseDIT256Radix16Complex64},
		{"Size512", 512, forwardDIT512Complex64, inverseDIT512Complex64},
		{"Size512/Radix8", 512, forwardDIT512Radix8Complex64, inverseDIT512Radix8Complex64},
		{"Size512/Radix4Then2", 512, forwardDIT512Radix4Then2Complex64, inverseDIT512Radix4Then2Complex64},
		{"Size512/Radix16x32", 512, forwardDIT512Mixed16x32Complex64, inverseDIT512Mixed16x32Complex64},
		{"Size1024/Radix4", 1024, forwardDIT1024Radix4Complex64, inverseDIT1024Radix4Complex64},
		{"Size1024/Radix32x32", 1024, forwardDIT1024Mixed32x32Complex64, inverseDIT1024Mixed32x32Complex64},
	}

	for _, testCase := range cases {
		b.Run(testCase.name+"/Forward", func(b *testing.B) {
			runBenchComplex64(b, testCase.n, testCase.forward)
		})
		b.Run(testCase.name+"/Inverse", func(b *testing.B) {
			runBenchComplex64(b, testCase.n, testCase.inverse)
		})
	}
}

// BenchmarkDITComplex128 benchmarks all Go DIT kernels for complex128.
func BenchmarkDITComplex128(b *testing.B) {
	cases := []benchCase128{
		{"Size4/Radix4", 4, forwardDIT4Radix4Complex128, inverseDIT4Radix4Complex128},
		{"Size8/Radix2", 8, forwardDIT8Radix2Complex128, inverseDIT8Radix2Complex128},
		{"Size8/Radix4", 8, forwardDIT8Radix4Complex128, inverseDIT8Radix4Complex128},
		{"Size8/Radix8", 8, forwardDIT8Radix8Complex128, inverseDIT8Radix8Complex128},
		{"Size16/Radix2", 16, forwardDIT16Radix2Complex128, inverseDIT16Radix2Complex128},
		{"Size16/Radix4", 16, forwardDIT16Radix4Complex128, inverseDIT16Radix4Complex128},
		{"Size32/Radix2", 32, forwardDIT32Complex128, inverseDIT32Complex128},
		{"Size32/Radix4x4x2", 32, forwardDIT32Radix4Then2Complex128, inverseDIT32Radix4Then2Complex128},
		{"Size64/Radix2", 64, forwardDIT64Complex128, inverseDIT64Complex128},
		{"Size64/Radix4", 64, forwardDIT64Radix4Complex128, inverseDIT64Radix4Complex128},
		{"Size128", 128, forwardDIT128Complex128, inverseDIT128Complex128},
		{"Size256/Radix2", 256, forwardDIT256Complex128, inverseDIT256Complex128},
		{"Size256/Radix4", 256, forwardDIT256Radix4Complex128, inverseDIT256Radix4Complex128},
		{"Size256/Radix16", 256, forwardDIT256Radix16Complex128, inverseDIT256Radix16Complex128},
		{"Size512", 512, forwardDIT512Complex128, inverseDIT512Complex128},
		{"Size512/Radix8", 512, forwardDIT512Radix8Complex128, inverseDIT512Radix8Complex128},
		{"Size512/Radix4Then2", 512, forwardDIT512Radix4Then2Complex128, inverseDIT512Radix4Then2Complex128},
		{"Size512/Radix16x32", 512, forwardDIT512Mixed16x32Complex128, inverseDIT512Mixed16x32Complex128},
		{"Size1024/Radix4", 1024, forwardDIT1024Radix4Complex128, inverseDIT1024Radix4Complex128},
		{"Size1024/Radix32x32", 1024, forwardDIT1024Mixed32x32Complex128, inverseDIT1024Mixed32x32Complex128},
	}

	for _, testCase := range cases {
		b.Run(testCase.name+"/Forward", func(b *testing.B) {
			runBenchComplex128(b, testCase.n, testCase.forward)
		})
		b.Run(testCase.name+"/Inverse", func(b *testing.B) {
			runBenchComplex128(b, testCase.n, testCase.inverse)
		})
	}
}

func runBenchComplex64(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex64) bool) {
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

	for b.Loop() {
		kernel(dst, src, twiddle, scratch)
	}
}

func runBenchComplex128(b *testing.B, n int, kernel func(dst, src, twiddle, scratch []complex128) bool) {
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

	for b.Loop() {
		kernel(dst, src, twiddle, scratch)
	}
}
