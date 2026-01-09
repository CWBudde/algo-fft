package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardDIT8Radix4Complex64 tests the size-8 radix-4 forward kernel.
func TestForwardDIT8Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xABCDEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size8Tol64)
}

// TestInverseDIT8Radix4Complex64 tests the size-8 radix-4 inverse kernel.
func TestInverseDIT8Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xFEDCBA)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex64 failed")
	}

	if !inverseDIT8Radix4Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT8Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size8Tol64)
}

// TestForwardDIT8Radix4Complex128 tests the size-8 radix-4 forward kernel (complex128).
func TestForwardDIT8Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0x1234)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size8Tol128)
}

// TestInverseDIT8Radix4Complex128 tests the size-8 radix-4 inverse kernel (complex128).
func TestInverseDIT8Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0x4321)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex128 failed")
	}

	if !inverseDIT8Radix4Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT8Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size8Tol128)
}

// TestRoundTripDIT8Radix4Complex64 tests forward then inverse returns original.
func TestRoundTripDIT8Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex64 failed")
	}

	if !inverseDIT8Radix4Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT8Radix4Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size8Tol64)
}

// TestRoundTripDIT8Radix4Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT8Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex128 failed")
	}

	if !inverseDIT8Radix4Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT8Radix4Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size8Tol128)
}
