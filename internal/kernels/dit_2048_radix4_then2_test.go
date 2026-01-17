package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size2048Radix4Then2Tol64  = 1e-4
	size2048Radix4Then2Tol128 = 1e-10
)

// TestForwardDIT2048Radix4Then2Complex64 tests the size-2048 forward radix-4-then-2 kernel.
func TestForwardDIT2048Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 2048

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT2048Radix4Then2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT2048Radix4Then2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size2048Radix4Then2Tol64)
}

// TestInverseDIT2048Radix4Then2Complex64 tests the size-2048 inverse radix-4-then-2 kernel.
func TestInverseDIT2048Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 2048

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT2048Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT2048Radix4Then2Complex64 failed")
	}

	if !inverseDIT2048Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT2048Radix4Then2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size2048Radix4Then2Tol64)
}

// TestRoundTripDIT2048Radix4Then2Complex64 tests forward then inverse returns original.
func TestRoundTripDIT2048Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 2048

	src := randomComplex64(n, 0xFEEDFACE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT2048Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT2048Radix4Then2Complex64 failed")
	}

	if !inverseDIT2048Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT2048Radix4Then2Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size2048Radix4Then2Tol64)
}

// TestForwardDIT2048Radix4Then2Complex128 tests the size-2048 forward radix-4-then-2 kernel (complex128).
func TestForwardDIT2048Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 2048

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT2048Radix4Then2Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT2048Radix4Then2Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size2048Radix4Then2Tol128)
}

// TestInverseDIT2048Radix4Then2Complex128 tests the size-2048 inverse radix-4-then-2 kernel (complex128).
func TestInverseDIT2048Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 2048

	src := randomComplex128(n, 0xDEADCAFE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT2048Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT2048Radix4Then2Complex128 failed")
	}

	if !inverseDIT2048Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT2048Radix4Then2Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size2048Radix4Then2Tol128)
}

// TestRoundTripDIT2048Radix4Then2Complex128 tests forward then inverse returns original.
func TestRoundTripDIT2048Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 2048

	src := randomComplex128(n, 0xCAFED00D)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT2048Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT2048Radix4Then2Complex128 failed")
	}

	if !inverseDIT2048Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT2048Radix4Then2Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size2048Radix4Then2Tol128)
}
