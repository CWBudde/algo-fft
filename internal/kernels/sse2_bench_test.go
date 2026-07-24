//go:build amd64 && !purego

package kernels

import (
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// BenchmarkSSE2Complex64 benchmarks SSE2/SSE3 kernels for complex64.
func BenchmarkSSE2Complex64(b *testing.B) {
	requireSSE3(b)

	cases := []benchCase64{
		{"Size4/Radix4", 4, amd64.ForwardSSE2Size4Radix4Complex64Asm, amd64.InverseSSE2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, amd64.ForwardSSE3Size8Radix2Complex64Asm, amd64.InverseSSE3Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, amd64.ForwardSSE3Size8Radix4Complex64Asm, amd64.InverseSSE3Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, amd64.ForwardSSE3Size8Radix8Complex64Asm, amd64.InverseSSE3Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, amd64.ForwardSSE3Size16Radix2Complex64Asm, amd64.InverseSSE3Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, amd64.ForwardSSE3Size16Radix4Complex64Asm, amd64.InverseSSE3Size16Radix4Complex64Asm},
		{"Size16/Radix16", 16, amd64.ForwardSSE3Size16Radix16Complex64Asm, amd64.InverseSSE3Size16Radix16Complex64Asm},
		{"Size64/Radix4", 64, amd64.ForwardSSE3Size64Radix4Complex64Asm, amd64.InverseSSE3Size64Radix4Complex64Asm},
		{"Size128/Radix4Then2", 128, amd64.ForwardSSE3Size128Radix4Then2Complex64Asm, amd64.InverseSSE3Size128Radix4Then2Complex64Asm},
		{"Size512/Radix2", 512, amd64.ForwardSSE3Size512Radix2Complex64Asm, amd64.InverseSSE3Size512Radix2Complex64Asm},
		{"Size512/Radix4Then2", 512, amd64.ForwardSSE3Size512Radix4Then2Complex64Asm, amd64.InverseSSE3Size512Radix4Then2Complex64Asm},
		{"Size1024/Radix4", 1024, amd64.ForwardSSE3Size1024Radix4Complex64Asm, amd64.InverseSSE3Size1024Radix4Complex64Asm},
		{"Size2048/Radix4Then2", 2048, amd64.ForwardSSE3Size2048Radix4Then2Complex64Asm, amd64.InverseSSE3Size2048Radix4Then2Complex64Asm},
	}

	for _, testCase := range cases {
		b.Run(testCase.name+"/Forward", func(b *testing.B) {
			if testCase.forward == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, testCase.n, testCase.forward)
		})
		b.Run(testCase.name+"/Inverse", func(b *testing.B) {
			if testCase.inverse == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, testCase.n, testCase.inverse)
		})
	}
}

// BenchmarkSSE2Complex128 benchmarks SSE2 kernels for complex128.
func BenchmarkSSE2Complex128(b *testing.B) {
	requireSSE3(b)

	cases := []benchCase128{
		{"Size256/Radix2", 256, amd64.ForwardSSE2Size256Radix2Complex128Asm, amd64.InverseSSE2Size256Radix2Complex128Asm},
		{"Size256/Radix4", 256, amd64.ForwardSSE2Size256Radix4Complex128Asm, amd64.InverseSSE2Size256Radix4Complex128Asm},
		{"Size512/Radix2", 512, amd64.ForwardSSE2Size512Radix2Complex128Asm, amd64.InverseSSE2Size512Radix2Complex128Asm},
		{"Size512/Radix4Then2", 512, amd64.ForwardSSE2Size512Radix4Then2Complex128Asm, amd64.InverseSSE2Size512Radix4Then2Complex128Asm},
		{"Size1024/Radix4", 1024, amd64.ForwardSSE2Size1024Radix4Complex128Asm, amd64.InverseSSE2Size1024Radix4Complex128Asm},
		{"Size2048/Radix4Then2", 2048, amd64.ForwardSSE2Size2048Radix4Then2Complex128Asm, amd64.InverseSSE2Size2048Radix4Then2Complex128Asm},
	}

	for _, testCase := range cases {
		b.Run(testCase.name+"/Forward", func(b *testing.B) {
			if testCase.forward == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex128(b, testCase.n, testCase.forward)
		})
		b.Run(testCase.name+"/Inverse", func(b *testing.B) {
			if testCase.inverse == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex128(b, testCase.n, testCase.inverse)
		})
	}
}
