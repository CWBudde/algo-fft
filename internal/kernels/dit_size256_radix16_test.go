package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardDIT256Radix16Complex64 tests the size-256 radix-16 forward kernel.
func TestForwardDIT256Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 256

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n) // Radix-16 uses binary bit-reversal

	if !forwardDIT256Radix16Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Radix16Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size256Tol64)
}

// TestInverseDIT256Radix16Complex64 tests the size-256 radix-16 inverse kernel.
func TestInverseDIT256Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 256

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT256Radix16Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Radix16Complex64 failed")
	}

	if !inverseDIT256Radix16Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT256Radix16Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size256Tol64)
}

// TestForwardDIT256Radix16Complex128 tests the size-256 radix-16 forward kernel (complex128).
func TestForwardDIT256Radix16Complex128(t *testing.T) {
	t.Parallel()

	const n = 256

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT256Radix16Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Radix16Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size256Tol128)
}

// TestInverseDIT256Radix16Complex128 tests the size-256 radix-16 inverse kernel (complex128).
func TestInverseDIT256Radix16Complex128(t *testing.T) {
	t.Parallel()

	const n = 256

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT256Radix16Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Radix16Complex128 failed")
	}

	if !inverseDIT256Radix16Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT256Radix16Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size256Tol128)
}

func BenchmarkForwardDIT256Radix16Complex64(b *testing.B) {
	const n = 256

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	b.ResetTimer()

	for range b.N {
		forwardDIT256Radix16Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkForwardDIT256Radix16Complex128(b *testing.B) {
	const n = 256

	src := randomComplex128(n, 0xDEADBEEF)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	for b.Loop() {
		forwardDIT256Radix16Complex128(dst, src, twiddle, scratch, bitrev)
	}
}
