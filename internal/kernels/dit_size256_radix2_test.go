package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size256Tol64  = 1e-4
	size256Tol128 = 1e-10
)

// TestForwardDIT256Complex64 tests the size-256 radix-2 forward kernel
func TestForwardDIT256Complex64(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size256Tol64)
}

// TestInverseDIT256Complex64 tests the size-256 radix-2 inverse kernel
func TestInverseDIT256Complex64(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Complex64 failed")
	}

	if !inverseDIT256Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT256Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size256Tol64)
}

// TestForwardDIT256Complex128 tests the size-256 radix-2 forward kernel (complex128)
func TestForwardDIT256Complex128(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size256Tol128)
}

// TestInverseDIT256Complex128 tests the size-256 radix-2 inverse kernel (complex128)
func TestInverseDIT256Complex128(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT256Complex128 failed")
	}

	if !inverseDIT256Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT256Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size256Tol128)
}
