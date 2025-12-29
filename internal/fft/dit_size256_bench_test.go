package fft

import "testing"

// Benchmarks comparing radix-2 and radix-4 implementations for size-256

// BenchmarkDIT256Radix2ForwardComplex64 benchmarks radix-2 forward transform
func BenchmarkDIT256Radix2ForwardComplex64(b *testing.B) {
	const n = 256
	src := randomComplex64(n, 0x1234)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // 8 bytes per complex64
	b.ResetTimer()

	for range b.N {
		forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

// BenchmarkDIT256Radix4ForwardComplex64 benchmarks radix-4 forward transform
func BenchmarkDIT256Radix4ForwardComplex64(b *testing.B) {
	const n = 256
	src := randomComplex64(n, 0x1234)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // 8 bytes per complex64
	b.ResetTimer()

	for range b.N {
		forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

// BenchmarkDIT256Radix2InverseComplex64 benchmarks radix-2 inverse transform
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

// BenchmarkDIT256Radix4InverseComplex64 benchmarks radix-4 inverse transform
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

// BenchmarkDIT256Radix2ForwardComplex128 benchmarks radix-2 forward transform for complex128
func BenchmarkDIT256Radix2ForwardComplex128(b *testing.B) {
	const n = 256
	src := randomComplex128(n, 0x9ABC)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16)) // 16 bytes per complex128
	b.ResetTimer()

	for range b.N {
		forwardDIT256Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

// BenchmarkDIT256Radix4ForwardComplex128 benchmarks radix-4 forward transform for complex128
func BenchmarkDIT256Radix4ForwardComplex128(b *testing.B) {
	const n = 256
	src := randomComplex128(n, 0x9ABC)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16)) // 16 bytes per complex128
	b.ResetTimer()

	for range b.N {
		forwardDIT256Radix4Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

// BenchmarkDIT256Radix2InverseComplex128 benchmarks radix-2 inverse transform for complex128
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

// BenchmarkDIT256Radix4InverseComplex128 benchmarks radix-4 inverse transform for complex128
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
