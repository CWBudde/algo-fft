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
		{"Size16/Radix16", 16, mathpkg.ComputeIdentityIndices, amd64.ForwardSSE2Size16Radix16Complex64Asm, nil},
		{"Size64/Radix4", 64, mathpkg.ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size64Radix4Complex64Asm, amd64.InverseSSE2Size64Radix4Complex64Asm},
		{"Size128/Radix4", 128, mathpkg.ComputeBitReversalIndices, amd64.ForwardSSE2Size128Radix4Complex64Asm, amd64.InverseSSE2Size128Radix4Complex64Asm},
	}

	for _, tc := range cases {
		b.Run(tc.name+"/Forward", func(b *testing.B) {
			if tc.forward == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, tc.n, tc.bitrev, tc.forward)
		})
		b.Run(tc.name+"/Inverse", func(b *testing.B) {
			if tc.inverse == nil {
				b.Skip("Not implemented")
			}
			runBenchComplex64(b, tc.n, tc.bitrev, tc.inverse)
		})
	}
}
