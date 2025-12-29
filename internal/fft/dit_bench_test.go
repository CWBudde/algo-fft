package fft

import "testing"

func BenchmarkDIT8_Specialized_Complex64(b *testing.B) {
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

func BenchmarkDIT8_Generic_Complex64(b *testing.B) {
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
		ditForward[complex64](dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT8_Specialized_Complex128(b *testing.B) {
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

func BenchmarkDIT8_Generic_Complex128(b *testing.B) {
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
		ditForward[complex128](dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_Specialized_Complex64(b *testing.B) {
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

func BenchmarkDIT16_Generic_Complex64(b *testing.B) {
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
		ditForward[complex64](dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_Specialized_Complex128(b *testing.B) {
	const n = 16
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
		forwardDIT16Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16_Generic_Complex128(b *testing.B) {
	const n = 16
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
		ditForward[complex128](dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT32_Specialized_Complex64(b *testing.B) {
	const n = 32
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
		forwardDIT32Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT32_Generic_Complex64(b *testing.B) {
	const n = 32
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

func BenchmarkDIT32_Specialized_Complex128(b *testing.B) {
	const n = 32
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
		forwardDIT32Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT32_Generic_Complex128(b *testing.B) {
	const n = 32
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
		ditForward[complex128](dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT64_Specialized_Complex64(b *testing.B) {
	const n = 64
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
		forwardDIT64Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT64_Generic_Complex64(b *testing.B) {
	const n = 64
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

func BenchmarkDIT256_Radix4_Complex64(b *testing.B) {
	const n = 256
	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for range b.N {
		forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
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

// Comprehensive radix-2 vs radix-4 benchmarks for size 16

func BenchmarkDIT16Radix2ForwardComplex64(b *testing.B) {
	const n = 16
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		forwardDIT16Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix4ForwardComplex64(b *testing.B) {
	const n = 16
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		forwardDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix2InverseComplex64(b *testing.B) {
	const n = 16
	src := randomComplex64(n, 0xCAFEBABE)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		inverseDIT16Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix4InverseComplex64(b *testing.B) {
	const n = 16
	src := randomComplex64(n, 0xCAFEBABE)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		inverseDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix2ForwardComplex128(b *testing.B) {
	const n = 16
	src := randomComplex128(n, 0xDEADBEEF)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		forwardDIT16Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix4ForwardComplex128(b *testing.B) {
	const n = 16
	src := randomComplex128(n, 0xDEADBEEF)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		forwardDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix2InverseComplex128(b *testing.B) {
	const n = 16
	src := randomComplex128(n, 0xCAFEBABE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		inverseDIT16Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT16Radix4InverseComplex128(b *testing.B) {
	const n = 16
	src := randomComplex128(n, 0xCAFEBABE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		inverseDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

// Comprehensive radix-2 vs radix-4 benchmarks for size 256

func BenchmarkDIT256Radix2ForwardComplex64(b *testing.B) {
	const n = 256
	src := randomComplex64(n, 0x1234)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix4ForwardComplex64(b *testing.B) {
	const n = 256
	src := randomComplex64(n, 0x1234)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix2InverseComplex64(b *testing.B) {
	const n = 256
	src := randomComplex64(n, 0x5678)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		inverseDIT256Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix4InverseComplex64(b *testing.B) {
	const n = 256
	src := randomComplex64(n, 0x5678)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for range b.N {
		inverseDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix2ForwardComplex128(b *testing.B) {
	const n = 256
	src := randomComplex128(n, 0x9ABC)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		forwardDIT256Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix4ForwardComplex128(b *testing.B) {
	const n = 256
	src := randomComplex128(n, 0x9ABC)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		forwardDIT256Radix4Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix2InverseComplex128(b *testing.B) {
	const n = 256
	src := randomComplex128(n, 0xDEF0)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		inverseDIT256Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkDIT256Radix4InverseComplex128(b *testing.B) {
	const n = 256
	src := randomComplex128(n, 0xDEF0)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for range b.N {
		inverseDIT256Radix4Complex128(dst, src, twiddle, scratch, bitrev)
	}
}
