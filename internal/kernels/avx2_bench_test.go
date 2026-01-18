//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// BenchmarkAVX2Complex64 benchmarks AVX2 kernels for complex64.
func BenchmarkAVX2Complex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, amd64.ForwardAVX2Size4Radix4Complex64Asm, amd64.InverseAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, amd64.ForwardAVX2Size8Radix2Complex64Asm, amd64.InverseAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, amd64.ForwardAVX2Size8Radix4Complex64Asm, amd64.InverseAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, amd64.ForwardAVX2Size8Radix8Complex64Asm, amd64.InverseAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, amd64.ForwardAVX2Size16Radix2Complex64Asm, amd64.InverseAVX2Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, amd64.ForwardAVX2Size16Radix4Complex64Asm, amd64.InverseAVX2Size16Radix4Complex64Asm},
		{"Size16/Radix16", 16, amd64.ForwardAVX2Size16Radix16Complex64Asm, amd64.InverseAVX2Size16Radix16Complex64Asm},
		{"Size32/Radix2", 32, amd64.ForwardAVX2Size32Radix2Complex64Asm, amd64.InverseAVX2Size32Radix2Complex64Asm},
		{"Size64/Radix2", 64, amd64.ForwardAVX2Size64Radix2Complex64Asm, amd64.InverseAVX2Size64Radix2Complex64Asm},
		{"Size64/Radix4", 64, amd64.ForwardAVX2Size64Radix4Complex64Asm, amd64.InverseAVX2Size64Radix4Complex64Asm},
		{"Size128/Radix4Then2", 128, amd64.ForwardAVX2Size128Radix4Then2Complex64Asm, amd64.InverseAVX2Size128Radix4Then2Complex64Asm},
		{"Size256/Radix2", 256, amd64.ForwardAVX2Size256Radix2Complex64Asm, amd64.InverseAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, amd64.ForwardAVX2Size256Radix4Complex64Asm, amd64.InverseAVX2Size256Radix4Complex64Asm},
		{"Size256/Radix16", 256, amd64.ForwardAVX2Size256Radix16Complex64Asm, amd64.InverseAVX2Size256Radix16Complex64Asm},
		{"Size512/Radix2", 512, amd64.ForwardAVX2Size512Radix2Complex64Asm, amd64.InverseAVX2Size512Radix2Complex64Asm},
		{"Size512/Radix8", 512, forwardAVX2Size512Radix8Complex64, inverseAVX2Size512Radix8Complex64},
		{"Size512/Radix4Then2", 512, amd64.ForwardAVX2Size512Radix4Then2Complex64Asm, amd64.InverseAVX2Size512Radix4Then2Complex64Asm},
		{"Size512/Radix16x32", 512, forwardAVX2Size512Radix16x32Complex64, inverseAVX2Size512Radix16x32Complex64},
		{"Size1024/Radix4", 1024, amd64.ForwardAVX2Size1024Radix4Complex64Asm, amd64.InverseAVX2Size1024Radix4Complex64Asm},
		{"Size1024/Radix32x32", 1024, amd64.ForwardAVX2Size1024Radix32x32Complex64Asm, amd64.InverseAVX2Size1024Radix32x32Complex64Asm},
	}

	for _, tc := range cases {
		b.Run(tc.name+"/Forward", func(b *testing.B) {
			if tc.forward == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, tc.n, tc.forward)
		})
		b.Run(tc.name+"/Inverse", func(b *testing.B) {
			if tc.inverse == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, tc.n, tc.inverse)
		})
	}
}

// BenchmarkAVX2Complex128 benchmarks AVX2 kernels for complex128.
func BenchmarkAVX2Complex128(b *testing.B) {
	cases := []benchCase128{
		{"Size256/Radix2", 256, amd64.ForwardAVX2Size256Radix2Complex128Asm, amd64.InverseAVX2Size256Radix2Complex128Asm},
		{"Size256/Radix4", 256, amd64.ForwardAVX2Size256Radix4Complex128Asm, amd64.InverseAVX2Size256Radix4Complex128Asm},
		{"Size256/Radix16", 256, amd64.ForwardAVX2Size256Radix16Complex128Asm, amd64.InverseAVX2Size256Radix16Complex128Asm},
		{"Size1024/Radix32x32", 1024, amd64.ForwardAVX2Size1024Radix32x32Complex128Asm, amd64.InverseAVX2Size1024Radix32x32Complex128Asm},
	}

	for _, tc := range cases {
		b.Run(tc.name+"/Forward", func(b *testing.B) {
			if tc.forward == nil {
				b.Skip("Not implemented")
			}
			if tc.n == 1024 {
				runBenchPreparedComplex128(b, tc.n, false, tc.forward)
				return
			}
			runBenchComplex128(b, tc.n, tc.forward)
		})
		b.Run(tc.name+"/Inverse", func(b *testing.B) {
			if tc.inverse == nil {
				b.Skip("Not implemented")
			}
			if tc.n == 1024 {
				runBenchPreparedComplex128(b, tc.n, true, tc.inverse)
				return
			}
			runBenchComplex128(b, tc.n, tc.inverse)
		})
	}
}

func runBenchPreparedComplex128(b *testing.B, n int, inverse bool, kernel func(dst, src, twiddle, scratch []complex128) bool) {
	b.Helper()

	src := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := make([]complex128, twiddleSize1024Radix32x32AVX2(n))
	prepareTwiddle1024Radix32x32AVX2(n, inverse, twiddle)

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
