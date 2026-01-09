package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size8Tol64  = 1e-4
	size8Tol128 = 1e-10
)

// TestForwardDIT8Radix2Complex64 tests the size-8 radix-2 forward kernel.
func TestForwardDIT8Radix2Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8Radix2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size8Tol64)
}

// TestInverseDIT8Radix2Complex64 tests the size-8 radix-2 inverse kernel.
func TestInverseDIT8Radix2Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8Radix2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix2Complex64 failed")
	}

	if !inverseDIT8Radix2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size8Tol64)
}

// TestForwardDIT8Radix2Complex128 tests the size-8 radix-2 forward kernel (complex128).
func TestForwardDIT8Radix2Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8Radix2Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix2Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size8Tol128)
}

// TestInverseDIT8Radix2Complex128 tests the size-8 radix-2 inverse kernel (complex128).
func TestInverseDIT8Radix2Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8Radix2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix2Complex128 failed")
	}

	if !inverseDIT8Radix2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix2Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size8Tol128)
}

// TestRoundTripDIT8Radix2Complex64 tests forward then inverse returns original.
func TestRoundTripDIT8Radix2Complex64(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8Radix2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix2Complex64 failed")
	}

	if !inverseDIT8Radix2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix2Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size8Tol64)
}

// TestRoundTripDIT8Radix2Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT8Radix2Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8Radix2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8Radix2Complex128 failed")
	}

	if !inverseDIT8Radix2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8Radix2Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size8Tol128)
}
