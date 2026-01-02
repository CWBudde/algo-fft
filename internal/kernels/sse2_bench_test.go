//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// BenchmarkSSE2Complex64 benchmarks SSE2 kernels for complex64.
func BenchmarkSSE2Complex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size4Radix4Complex64Asm, amd64.InverseSSE2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, amd64.ForwardSSE2Size8Radix2Complex64Asm, amd64.InverseSSE2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, ComputeBitReversalIndices, amd64.ForwardSSE2Size8Radix4Complex64Asm, amd64.InverseSSE2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, ComputeBitReversalIndices, amd64.ForwardSSE2Size8Radix8Complex64Asm, amd64.InverseSSE2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, ComputeBitReversalIndices, amd64.ForwardSSE2Size16Radix2Complex64Asm, amd64.InverseSSE2Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size16Radix4Complex64Asm, amd64.InverseSSE2Size16Radix4Complex64Asm},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, amd64.ForwardSSE2Size64Radix4Complex64Asm, amd64.InverseSSE2Size64Radix4Complex64Asm},
		{"Size128/Radix4", 128, ComputeBitReversalIndices, amd64.ForwardSSE2Size128Radix4Complex64Asm, amd64.InverseSSE2Size128Radix4Complex64Asm},
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
