//go:build amd64 && fft_asm && !purego

package fft

import "testing"

func BenchmarkDIT8_InverseGo_Complex64(b *testing.B) {
	const n = 8
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		inverseDIT8Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT8_AVX2_Complex64(b *testing.B) {
	const n = 8
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT8_InverseAVX2_Complex64(b *testing.B) {
	const n = 8
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_ForwardGo_Complex64(b *testing.B) {
	const n = 16
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		forwardDIT16Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_InverseGo_Complex64(b *testing.B) {
	const n = 16
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		inverseDIT16Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_ForwardAVX2_Complex64(b *testing.B) {
	const n = 16
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_InverseAVX2_Complex64(b *testing.B) {
	const n = 16
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}
