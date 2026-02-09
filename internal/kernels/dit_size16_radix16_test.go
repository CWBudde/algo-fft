package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestForwardDIT16Radix16Complex64 tests the size-16 radix-16 forward kernel.
func TestForwardDIT16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0x1234BEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16Radix16Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT16Radix16Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size16Tol64)
}

// TestInverseDIT16Radix16Complex64 tests the size-16 radix-16 inverse kernel.
func TestInverseDIT16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0x5678CAFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16Radix16Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16Radix16Complex64 failed")
	}

	if !inverseDIT16Radix16Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16Radix16Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size16Tol64)
}

// TestForwardDIT16Radix16Complex128 tests the size-16 radix-16 forward kernel (complex128).
func TestForwardDIT16Radix16Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex128(n, 0xBEEF1234)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16Radix16Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT16Radix16Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size16Tol128)
}

// TestInverseDIT16Radix16Complex128 tests the size-16 radix-16 inverse kernel (complex128).
func TestInverseDIT16Radix16Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex128(n, 0xCAFE5678)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16Radix16Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16Radix16Complex128 failed")
	}

	if !inverseDIT16Radix16Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16Radix16Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size16Tol128)
}

// TestRoundTripDIT16Radix16Complex64 tests forward then inverse returns original.
func TestRoundTripDIT16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0x0BADF00D)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16Radix16Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16Radix16Complex64 failed")
	}

	if !inverseDIT16Radix16Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16Radix16Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size16Tol64)
}

// TestRoundTripDIT16Radix16Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT16Radix16Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex128(n, 0xF00D0BAD)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16Radix16Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16Radix16Complex128 failed")
	}

	if !inverseDIT16Radix16Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16Radix16Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size16Tol128)
}
