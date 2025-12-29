//go:build amd64 && fft_asm && !purego

package fft

import "testing"

type benchCaseAVX2 struct {
	name    string
	n       int
	bitrev  func(int) []int
	forward func(dst, src, twiddle, scratch []complex64, bitrev []int) bool
	inverse func(dst, src, twiddle, scratch []complex64, bitrev []int) bool
}

func BenchmarkDITAVX2Complex64(b *testing.B) {
	cases := []benchCaseAVX2{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex64Asm, inverseAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, forwardAVX2Size8Radix2Complex64Asm, inverseAVX2Size8Radix2Complex64Asm},
		{"Size8/MixedRadix", 8, ComputeBitReversalIndices, forwardAVX2Size8Radix4Complex64Asm, inverseAVX2Size8Radix4Complex64Asm},
		{"Size16/Radix2", 16, ComputeBitReversalIndices, forwardAVX2Size16Complex64Asm, inverseAVX2Size16Complex64Asm},
		{"Size16/Radix4", 16, ComputeBitReversalIndicesRadix4, forwardAVX2Size16Radix4Complex64Asm, inverseAVX2Size16Radix4Complex64Asm},
		{"Size32/Radix2", 32, ComputeBitReversalIndices, forwardAVX2Size32Complex64Asm, inverseAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, forwardAVX2Size64Complex64Asm, inverseAVX2Size64Complex64Asm},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex64Asm, inverseAVX2Size64Radix4Complex64Asm},
		{"Size128/Radix2", 128, ComputeBitReversalIndices, forwardAVX2Size128Complex64Asm, inverseAVX2Size128Complex64Asm},
		{"Size256/Radix2", 256, ComputeBitReversalIndices, forwardAVX2Size256Radix2Complex64Asm, inverseAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, ComputeBitReversalIndicesRadix4, forwardAVX2Size256Radix4Complex64Asm, inverseAVX2Size256Radix4Complex64Asm},
	}

	for _, tc := range cases {
		tc := tc
		b.Run(tc.name+"/Forward", func(b *testing.B) {
			runBenchAVX2Complex64(b, tc.n, tc.bitrev, tc.forward)
		})
		b.Run(tc.name+"/Inverse", func(b *testing.B) {
			runBenchAVX2Complex64(b, tc.n, tc.bitrev, tc.inverse)
		})
	}
}

func runBenchAVX2Complex64(b *testing.B, n int, bitrev func(int) []int, kernel func(dst, src, twiddle, scratch []complex64, bitrev []int) bool) {
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

	for range b.N {
		kernel(dst, src, twiddle, scratch, br)
	}
}
