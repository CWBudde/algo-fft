package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size16Tol64  = 1e-4
	size16Tol128 = 1e-10
)

// TestForwardDIT16Complex64 tests the size-16 radix-2 forward kernel.
func TestForwardDIT16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT16Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size16Tol64)
}

// TestInverseDIT16Complex64 tests the size-16 radix-2 inverse kernel.
func TestInverseDIT16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT16Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex64 failed")
	}

	if !inverseDIT16Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size16Tol64)
}

// TestForwardDIT16Complex128 tests the size-16 radix-2 forward kernel (complex128).
func TestForwardDIT16Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT16Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size16Tol128)
}

// TestInverseDIT16Complex128 tests the size-16 radix-2 inverse kernel (complex128).
func TestInverseDIT16Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT16Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex128 failed")
	}

	if !inverseDIT16Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size16Tol128)
}

// TestRoundTripDIT16Complex64 tests forward then inverse returns original.
func TestRoundTripDIT16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT16Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex64 failed")
	}

	if !inverseDIT16Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size16Tol64)
}

// TestRoundTripDIT16Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT16Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !forwardDIT16Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex128 failed")
	}

	if !inverseDIT16Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size16Tol128)
}
