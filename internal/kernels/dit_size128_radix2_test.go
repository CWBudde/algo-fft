package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size128Tol64  = 1e-4
	size128Tol128 = 1e-10
)

// TestForwardDIT128Complex64 tests the size-128 forward kernel.
func TestForwardDIT128Complex64(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT128Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT128Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size128Tol64)
}

// TestInverseDIT128Complex64 tests the size-128 inverse kernel.
func TestInverseDIT128Complex64(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT128Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT128Complex64 failed")
	}

	if !inverseDIT128Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT128Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size128Tol64)
}

// TestForwardDIT128Complex128 tests the size-128 forward kernel (complex128).
func TestForwardDIT128Complex128(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT128Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT128Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size128Tol128)
}

// TestInverseDIT128Complex128 tests the size-128 inverse kernel (complex128).
func TestInverseDIT128Complex128(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT128Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT128Complex128 failed")
	}

	if !inverseDIT128Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT128Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size128Tol128)
}
