package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestForwardDIT8Radix8Complex64 tests the size-8 radix-8 forward kernel.
func TestForwardDIT8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0x12345678)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8Radix8Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix8Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size8Tol64)
}

// TestInverseDIT8Radix8Complex64 tests the size-8 radix-8 inverse kernel.
func TestInverseDIT8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0x87654321)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8Radix8Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix8Complex64 failed")
	}

	if !inverseDIT8Radix8Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix8Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size8Tol64)
}

// TestForwardDIT8Radix8Complex128 tests the size-8 radix-8 forward kernel (complex128).
func TestForwardDIT8Radix8Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xDEADCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8Radix8Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix8Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size8Tol128)
}

// TestInverseDIT8Radix8Complex128 tests the size-8 radix-8 inverse kernel (complex128).
func TestInverseDIT8Radix8Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xCAFEDEAD)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8Radix8Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix8Complex128 failed")
	}

	if !inverseDIT8Radix8Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix8Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size8Tol128)
}

// TestRoundTripDIT8Radix8Complex64 tests forward then inverse returns original.
func TestRoundTripDIT8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8Radix8Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix8Complex64 failed")
	}

	if !inverseDIT8Radix8Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix8Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size8Tol64)
}

// TestRoundTripDIT8Radix8Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT8Radix8Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8Radix8Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix8Complex128 failed")
	}

	if !inverseDIT8Radix8Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix8Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size8Tol128)
}
