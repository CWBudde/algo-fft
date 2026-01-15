//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// BenchmarkSSE2Complex64 benchmarks SSE2/SSE3 kernels for complex64.
func BenchmarkSSE2Complex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, amd64.ForwardSSE2Size4Radix4Complex64Asm, amd64.InverseSSE2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, amd64.ForwardSSE3Size8Radix2Complex64Asm, amd64.InverseSSE3Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, amd64.ForwardSSE3Size8Radix4Complex64Asm, amd64.InverseSSE3Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, amd64.ForwardSSE3Size8Radix8Complex64Asm, amd64.InverseSSE3Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, amd64.ForwardSSE3Size16Radix2Complex64Asm, amd64.InverseSSE3Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, amd64.ForwardSSE3Size16Radix4Complex64Asm, amd64.InverseSSE3Size16Radix4Complex64Asm},
		{"Size16/Radix16", 16, amd64.ForwardSSE3Size16Radix16Complex64Asm, amd64.InverseSSE3Size16Radix16Complex64Asm},
		{"Size64/Radix4", 64, amd64.ForwardSSE3Size64Radix4Complex64Asm, amd64.InverseSSE3Size64Radix4Complex64Asm},
		{"Size128/Mixed24", 128, amd64.ForwardSSE3Size128Mixed24Complex64Asm, amd64.InverseSSE3Size128Mixed24Complex64Asm},
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
