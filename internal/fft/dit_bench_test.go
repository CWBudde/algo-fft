package fft

import "testing"

func BenchmarkDIT256_Specialized_Complex64(b *testing.B) {
	const n = 256
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
		forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256_Generic_Complex64(b *testing.B) {
	const n = 256
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
		ditForward[complex64](dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT512_Specialized_Complex64(b *testing.B) {
	const n = 512
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
		forwardDIT512Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT512_Generic_Complex64(b *testing.B) {
	const n = 512
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
		ditForward[complex64](dst, src, twiddle, scratch, bitrev)
	}
}
