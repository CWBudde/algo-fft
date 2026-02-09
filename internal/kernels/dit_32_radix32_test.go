package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestForwardDIT32Radix32Complex64 tests the size-32 radix-32 forward kernel.
func TestForwardDIT32Radix32Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0x1234ABCD)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32Radix32Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix32Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size32Tol64)
}

// TestInverseDIT32Radix32Complex64 tests the size-32 radix-32 inverse kernel.
func TestInverseDIT32Radix32Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0x5678DCBA)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32Radix32Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix32Complex64 failed")
	}

	if !inverseDIT32Radix32Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix32Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size32Tol64)
}

// TestRoundTripDIT32Radix32Complex64 tests forward then inverse returns original.
func TestRoundTripDIT32Radix32Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0x9ABCDEF0)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32Radix32Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix32Complex64 failed")
	}

	if !inverseDIT32Radix32Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix32Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size32Tol64)
}

// TestForwardDIT32Radix32Complex128 tests the size-32 radix-32 forward kernel (complex128).
func TestForwardDIT32Radix32Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x13579BDF)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32Radix32Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix32Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size32Tol128)
}

// TestInverseDIT32Radix32Complex128 tests the size-32 radix-32 inverse kernel (complex128).
func TestInverseDIT32Radix32Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x2468ACE0)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32Radix32Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix32Complex128 failed")
	}

	if !inverseDIT32Radix32Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix32Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size32Tol128)
}

// TestRoundTripDIT32Radix32Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT32Radix32Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x0F1E2D3C)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32Radix32Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix32Complex128 failed")
	}

	if !inverseDIT32Radix32Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix32Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size32Tol128)
}
