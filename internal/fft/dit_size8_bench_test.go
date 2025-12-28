package fft

import "testing"

func BenchmarkDIT8_Forward_Complex64(b *testing.B) {
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
		forwardDIT8Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT8_Inverse_Complex64(b *testing.B) {
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

func BenchmarkDIT8_Forward_Complex128(b *testing.B) {
	const n = 8
	src := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float64(i), float64(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 16))

	for range b.N {
		forwardDIT8Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT8_Inverse_Complex128(b *testing.B) {
	const n = 8
	src := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	for i := range src {
		src[i] = complex(float64(i), float64(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 16))

	for range b.N {
		inverseDIT8Complex128(dst, src, twiddle, scratch, bitrev)
	}
}
