package fft

import (
	"testing"
)

// Benchmarks for size-16 DIT FFT implementations

// BenchmarkDIT16Radix2ForwardComplex64 benchmarks the radix-2 forward transform
func BenchmarkDIT16Radix2ForwardComplex64(b *testing.B) {
	const n = 16
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // complex64 = 8 bytes
	b.ResetTimer()

	for range b.N {
		forwardDIT16Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

// BenchmarkDIT16Radix4ForwardComplex64 benchmarks the radix-4 forward transform
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

// BenchmarkDIT16Radix2InverseComplex64 benchmarks the radix-2 inverse transform
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

// BenchmarkDIT16Radix4InverseComplex64 benchmarks the radix-4 inverse transform
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

// BenchmarkDIT16Radix2ForwardComplex128 benchmarks the radix-2 forward transform (complex128)
func BenchmarkDIT16Radix2ForwardComplex128(b *testing.B) {
	const n = 16
	src := randomComplex128(n, 0xDEADBEEF)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16)) // complex128 = 16 bytes
	b.ResetTimer()

	for range b.N {
		forwardDIT16Complex128(dst, src, twiddle, scratch, bitrev)
	}
}

// BenchmarkDIT16Radix4ForwardComplex128 benchmarks the radix-4 forward transform (complex128)
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

// BenchmarkDIT16Radix2InverseComplex128 benchmarks the radix-2 inverse transform (complex128)
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

// BenchmarkDIT16Radix4InverseComplex128 benchmarks the radix-4 inverse transform (complex128)
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
