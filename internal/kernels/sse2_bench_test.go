//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// BenchmarkSSE2Complex64 benchmarks SSE2 kernels for complex64.
func BenchmarkSSE2Complex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, mathpkg.ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size4Radix4Complex64Asm, amd64.InverseSSE2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, mathpkg.ComputeBitReversalIndices, amd64.ForwardSSE2Size8Radix2Complex64Asm, amd64.InverseSSE2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, mathpkg.ComputeBitReversalIndices, amd64.ForwardSSE2Size8Radix4Complex64Asm, amd64.InverseSSE2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, amd64.ForwardSSE2Size8Radix8Complex64Asm, amd64.InverseSSE2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, mathpkg.ComputeBitReversalIndices, amd64.ForwardSSE2Size16Radix2Complex64Asm, amd64.InverseSSE2Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, mathpkg.ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size16Radix4Complex64Asm, amd64.InverseSSE2Size16Radix4Complex64Asm},
		{"Size16/Radix16", 16, mathpkg.ComputeIdentityIndices, amd64.ForwardSSE2Size16Radix16Complex64Asm, amd64.InverseSSE2Size16Radix16Complex64Asm},
		{"Size64/Radix4", 64, mathpkg.ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size64Radix4Complex64Asm, amd64.InverseSSE2Size64Radix4Complex64Asm},
		{"Size128/Mixed24", 128, mathpkg.ComputeBitReversalIndicesMixed24, amd64.ForwardSSE2Size128Mixed24Complex64Asm, amd64.InverseSSE2Size128Mixed24Complex64Asm},
	}

	for _, testCase := range cases {
		b.Run(testCase.name+"/Forward", func(b *testing.B) {
			if testCase.forward == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, testCase.n, testCase.bitrev, testCase.forward)
		})
		b.Run(testCase.name+"/Inverse", func(b *testing.B) {
			if testCase.inverse == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, testCase.n, testCase.bitrev, testCase.inverse)
		})
	}
}
