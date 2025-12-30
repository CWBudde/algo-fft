package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size512Tol64  = 1e-4
	size512Tol128 = 1e-10
)

// TestForwardDIT512Complex64 tests the size-512 forward kernel
func TestForwardDIT512Complex64(t *testing.T) {
	t.Parallel()

	const n = 512
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT512Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size512Tol64)
}

// TestInverseDIT512Complex64 tests the size-512 inverse kernel
func TestInverseDIT512Complex64(t *testing.T) {
	t.Parallel()

	const n = 512
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT512Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Complex64 failed")
	}

	if !inverseDIT512Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT512Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size512Tol64)
}

// TestForwardDIT512Complex128 tests the size-512 forward kernel (complex128)
func TestForwardDIT512Complex128(t *testing.T) {
	t.Parallel()

	const n = 512
	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT512Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size512Tol128)
}

// TestInverseDIT512Complex128 tests the size-512 inverse kernel (complex128)
func TestInverseDIT512Complex128(t *testing.T) {
	t.Parallel()

	const n = 512
	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT512Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Complex128 failed")
	}

	if !inverseDIT512Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT512Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size512Tol128)
}
