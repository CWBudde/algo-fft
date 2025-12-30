package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size32Tol64  = 1e-4
	size32Tol128 = 1e-10
)

// TestForwardDIT32Complex64 tests the size-32 forward kernel
func TestForwardDIT32Complex64(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size32Tol64)
}

// TestInverseDIT32Complex64 tests the size-32 inverse kernel
func TestInverseDIT32Complex64(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32Complex64 failed")
	}

	if !inverseDIT32Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT32Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size32Tol64)
}

// TestForwardDIT32Complex128 tests the size-32 forward kernel (complex128)
func TestForwardDIT32Complex128(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size32Tol128)
}

// TestInverseDIT32Complex128 tests the size-32 inverse kernel (complex128)
func TestInverseDIT32Complex128(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32Complex128 failed")
	}

	if !inverseDIT32Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT32Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size32Tol128)
}
