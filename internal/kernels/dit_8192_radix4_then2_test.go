package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size8192Radix4Then2Tol64  = 1e-3 // Slightly higher tolerance for larger sizes
	size8192Radix4Then2Tol128 = 1e-9
)

// TestForwardDIT8192Radix4Then2Complex64 tests the size-8192 forward radix-4-then-2 kernel.
func TestForwardDIT8192Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 8192

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8192Radix4Then2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size8192Radix4Then2Tol64)
}

// TestInverseDIT8192Radix4Then2Complex64 tests the size-8192 inverse radix-4-then-2 kernel.
func TestInverseDIT8192Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 8192

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8192Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex64 failed")
	}

	if !inverseDIT8192Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8192Radix4Then2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size8192Radix4Then2Tol64)
}

// TestRoundTripDIT8192Radix4Then2Complex64 tests forward then inverse returns original.
func TestRoundTripDIT8192Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 8192

	src := randomComplex64(n, 0xFEEDFACE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT8192Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex64 failed")
	}

	if !inverseDIT8192Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8192Radix4Then2Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size8192Radix4Then2Tol64)
}

// TestForwardDIT8192Radix4Then2Complex128 tests the size-8192 forward radix-4-then-2 kernel (complex128).
func TestForwardDIT8192Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 8192

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8192Radix4Then2Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size8192Radix4Then2Tol128)
}

// TestInverseDIT8192Radix4Then2Complex128 tests the size-8192 inverse radix-4-then-2 kernel (complex128).
func TestInverseDIT8192Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 8192

	src := randomComplex128(n, 0xDEADCAFE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8192Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex128 failed")
	}

	if !inverseDIT8192Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8192Radix4Then2Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size8192Radix4Then2Tol128)
}

// TestRoundTripDIT8192Radix4Then2Complex128 tests forward then inverse returns original.
func TestRoundTripDIT8192Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 8192

	src := randomComplex128(n, 0xCAFED00D)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT8192Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex128 failed")
	}

	if !inverseDIT8192Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT8192Radix4Then2Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size8192Radix4Then2Tol128)
}
