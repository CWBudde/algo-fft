//go:build 386 && asm && !purego

package kernels

import (
	"testing"

	x86 "github.com/MeKo-Christian/algo-fft/internal/asm/x86"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestForwardSSE3Size8Radix2Complex64_386(t *testing.T) {
	const n = 8
	src := randomComplex64(n, 0x12345678)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	want := make([]complex64, n)
	copy(want, src)
	forwardDIT8Complex64(want, want, twiddle, scratch)

	if !x86.ForwardSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE3Size8Radix2Complex64Asm failed")
	}

	assertComplex64Close(t, dst, want, 1e-5)
}

func TestInverseSSE3Size8Radix2Complex64_386(t *testing.T) {
	const n = 8
	src := randomComplex64(n, 0x87654321)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	want := make([]complex64, n)
	copy(want, src)
	inverseDIT8Complex64(want, want, twiddle, scratch)

	if !x86.InverseSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE3Size8Radix2Complex64Asm failed")
	}

	assertComplex64Close(t, dst, want, 1e-5)
}

func TestRoundTripSSE3Size8Radix2Complex64_386(t *testing.T) {
	const n = 8
	src := randomComplex64(n, 0xABCDEF)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)

	if !x86.ForwardSSE3Size8Radix2Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("Forward failed")
	}
	if !x86.InverseSSE3Size8Radix2Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("Inverse failed")
	}

	assertComplex64Close(t, inv, src, 1e-5)
}

func BenchmarkForwardSSE3Size8Radix2Complex64_386(b *testing.B) {
	const n = 8
	src := randomComplex64(n, 0x99999999)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16)) // 8 bytes per complex64, 2x for read+write
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		x86.ForwardSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkInverseSSE3Size8Radix2Complex64_386(b *testing.B) {
	const n = 8
	src := randomComplex64(n, 0x88888888)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		x86.InverseSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}
