//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// BenchmarkAVX2Complex64 benchmarks AVX2 kernels for complex64.
func BenchmarkAVX2Complex64(b *testing.B) {
	cases := []benchCase64{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, amd64.ForwardAVX2Size4Radix4Complex64Asm, amd64.InverseAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, amd64.ForwardAVX2Size8Radix2Complex64Asm, amd64.InverseAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, ComputeBitReversalIndices, amd64.ForwardAVX2Size8Radix4Complex64Asm, amd64.InverseAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, ComputeBitReversalIndices, amd64.ForwardAVX2Size8Radix8Complex64Asm, amd64.InverseAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, ComputeBitReversalIndices, amd64.ForwardAVX2Size16Complex64Asm, amd64.InverseAVX2Size16Complex64Asm},
		{"Size32/Radix2", 32, ComputeBitReversalIndices, amd64.ForwardAVX2Size32Complex64Asm, amd64.InverseAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, amd64.ForwardAVX2Size64Complex64Asm, amd64.InverseAVX2Size64Complex64Asm},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, amd64.ForwardAVX2Size64Radix4Complex64Asm, amd64.InverseAVX2Size64Radix4Complex64Asm},
		{"Size256/Radix2", 256, ComputeBitReversalIndices, amd64.ForwardAVX2Size256Radix2Complex64Asm, amd64.InverseAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, ComputeBitReversalIndicesRadix4, amd64.ForwardAVX2Size256Radix4Complex64Asm, amd64.InverseAVX2Size256Radix4Complex64Asm},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, amd64.ForwardAVX2Size512Radix2Complex64Asm, amd64.InverseAVX2Size512Radix2Complex64Asm},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, amd64.ForwardAVX2Size512Mixed24Complex64Asm, amd64.InverseAVX2Size512Mixed24Complex64Asm},
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
