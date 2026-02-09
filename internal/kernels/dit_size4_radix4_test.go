package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	size4Tol64  = 1e-4
	size4Tol128 = 1e-10
)

// TestForwardDIT4Radix4Complex64 tests the size-4 radix-4 forward kernel.
func TestForwardDIT4Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT4Radix4Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT4Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size4Tol64)
}

// TestInverseDIT4Radix4Complex64 tests the size-4 radix-4 inverse kernel.
func TestInverseDIT4Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT4Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4Radix4Complex64 failed")
	}

	if !inverseDIT4Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size4Tol64)
}

// TestForwardDIT4Radix4Complex128 tests the size-4 radix-4 forward kernel (complex128).
func TestForwardDIT4Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT4Radix4Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT4Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size4Tol128)
}

// TestInverseDIT4Radix4Complex128 tests the size-4 radix-4 inverse kernel (complex128).
func TestInverseDIT4Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT4Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4Radix4Complex128 failed")
	}

	if !inverseDIT4Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size4Tol128)
}

// TestRoundTripDIT4Radix4Complex64 tests forward then inverse returns original.
func TestRoundTripDIT4Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT4Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4Radix4Complex64 failed")
	}

	if !inverseDIT4Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4Radix4Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size4Tol64)
}

// TestRoundTripDIT4Radix4Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT4Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT4Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4Radix4Complex128 failed")
	}

	if !inverseDIT4Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4Radix4Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size4Tol128)
}
