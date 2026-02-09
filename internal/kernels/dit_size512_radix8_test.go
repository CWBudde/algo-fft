package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	size512Radix8Tol64  = 1e-4
	size512Radix8Tol128 = 1e-10
)

// TestForwardDIT512Radix8Complex64 tests the size-512 forward radix-8 kernel.
func TestForwardDIT512Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT512Radix8Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT512Radix8Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size512Radix8Tol64)
}

// TestInverseDIT512Radix8Complex64 tests the size-512 inverse radix-8 kernel.
func TestInverseDIT512Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT512Radix8Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT512Radix8Complex64 failed")
	}

	if !inverseDIT512Radix8Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT512Radix8Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size512Radix8Tol64)
}

// TestRoundTripDIT512Radix8Complex64 tests forward then inverse returns original.
func TestRoundTripDIT512Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex64(n, 0xFEEDFACE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT512Radix8Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT512Radix8Complex64 failed")
	}

	if !inverseDIT512Radix8Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT512Radix8Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size512Radix8Tol64)
}

// TestForwardDIT512Radix8Complex128 tests the size-512 forward radix-8 kernel (complex128).
func TestForwardDIT512Radix8Complex128(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT512Radix8Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT512Radix8Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size512Radix8Tol128)
}

// TestInverseDIT512Radix8Complex128 tests the size-512 inverse radix-8 kernel (complex128).
func TestInverseDIT512Radix8Complex128(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex128(n, 0xDEADCAFE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT512Radix8Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT512Radix8Complex128 failed")
	}

	if !inverseDIT512Radix8Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT512Radix8Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size512Radix8Tol128)
}

// TestRoundTripDIT512Radix8Complex128 tests forward then inverse returns original.
func TestRoundTripDIT512Radix8Complex128(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex128(n, 0xCAFED00D)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT512Radix8Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT512Radix8Complex128 failed")
	}

	if !inverseDIT512Radix8Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT512Radix8Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size512Radix8Tol128)
}
