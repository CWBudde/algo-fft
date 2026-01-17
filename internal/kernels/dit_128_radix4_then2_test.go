package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size128Radix4Then2Tol64  = 1e-4
	size128Radix4Then2Tol128 = 1e-10
)

// TestForwardDIT128Radix4Then2Complex64 tests the size-128 forward radix-4-then-2 kernel.
func TestForwardDIT128Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex64(n, 0x1234ABCD)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT128Radix4Then2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT128Radix4Then2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size128Radix4Then2Tol64)
}

// TestInverseDIT128Radix4Then2Complex64 tests the size-128 inverse radix-4-then-2 kernel.
func TestInverseDIT128Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex64(n, 0xDEADBEEF)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT128Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT128Radix4Then2Complex64 failed")
	}

	if !inverseDIT128Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT128Radix4Then2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size128Radix4Then2Tol64)
}

// TestRoundTripDIT128Radix4Then2Complex64 tests forward then inverse returns original.
func TestRoundTripDIT128Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex64(n, 0xBEEFCAFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT128Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT128Radix4Then2Complex64 failed")
	}

	if !inverseDIT128Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT128Radix4Then2Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size128Radix4Then2Tol64)
}

// TestForwardDIT128Radix4Then2Complex128 tests the size-128 forward radix-4-then-2 kernel (complex128).
func TestForwardDIT128Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex128(n, 0xFEEDFACE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT128Radix4Then2Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT128Radix4Then2Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size128Radix4Then2Tol128)
}

// TestInverseDIT128Radix4Then2Complex128 tests the size-128 inverse radix-4-then-2 kernel (complex128).
func TestInverseDIT128Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex128(n, 0xCAFED00D)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT128Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT128Radix4Then2Complex128 failed")
	}

	if !inverseDIT128Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT128Radix4Then2Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size128Radix4Then2Tol128)
}

// TestRoundTripDIT128Radix4Then2Complex128 tests forward then inverse returns original.
func TestRoundTripDIT128Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 128

	src := randomComplex128(n, 0x12345678)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT128Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT128Radix4Then2Complex128 failed")
	}

	if !inverseDIT128Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT128Radix4Then2Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size128Radix4Then2Tol128)
}
