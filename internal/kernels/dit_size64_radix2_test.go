package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	size64Tol64  = 1e-4
	size64Tol128 = 1e-10
)

// TestForwardDIT64Complex64 tests the size-64 forward kernel.
func TestForwardDIT64Complex64(t *testing.T) {
	t.Parallel()

	const n = 64

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT64Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT64Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size64Tol64)
}

// TestInverseDIT64Complex64 tests the size-64 inverse kernel.
func TestInverseDIT64Complex64(t *testing.T) {
	t.Parallel()

	const n = 64

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT64Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT64Complex64 failed")
	}

	if !inverseDIT64Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT64Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size64Tol64)
}

// TestForwardDIT64Complex128 tests the size-64 forward kernel (complex128).
func TestForwardDIT64Complex128(t *testing.T) {
	t.Parallel()

	const n = 64

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT64Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT64Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size64Tol128)
}

// TestInverseDIT64Complex128 tests the size-64 inverse kernel (complex128).
func TestInverseDIT64Complex128(t *testing.T) {
	t.Parallel()

	const n = 64

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT64Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT64Complex128 failed")
	}

	if !inverseDIT64Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT64Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size64Tol128)
}

// TestRoundTripDIT64Complex64 tests forward then inverse returns original.
func TestRoundTripDIT64Complex64(t *testing.T) {
	t.Parallel()

	const n = 64

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT64Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT64Complex64 failed")
	}

	if !inverseDIT64Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT64Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size64Tol64)
}

// TestRoundTripDIT64Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT64Complex128(t *testing.T) {
	t.Parallel()

	const n = 64

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT64Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT64Complex128 failed")
	}

	if !inverseDIT64Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT64Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size64Tol128)
}
