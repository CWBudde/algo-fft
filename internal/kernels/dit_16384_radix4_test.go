package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	size16384Tol64  = 1e-3
	size16384Tol128 = 2e-9 // Slightly relaxed due to accumulated floating-point errors in 7 stages
)

// TestForwardDIT16384Radix4Complex64 tests the size-16384 forward radix-4 kernel
// Size 16384 = 4^7, so this uses 7 radix-4 stages instead of 14 radix-2 stages.
func TestForwardDIT16384Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 16384

	skipNaiveReferenceIfSlow(t)

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16384Radix4Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size16384Tol64)
}

// TestInverseDIT16384Radix4Complex64 tests the size-16384 inverse radix-4 kernel.
func TestInverseDIT16384Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 16384

	skipNaiveReferenceIfSlow(t)

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16384Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex64 failed")
	}

	if !inverseDIT16384Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16384Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size16384Tol64)
}

// TestRoundTripDIT16384Radix4Complex64 tests forward then inverse returns original.
func TestRoundTripDIT16384Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 16384

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16384Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex64 failed")
	}

	if !inverseDIT16384Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16384Radix4Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size16384Tol64)
}

// TestForwardDIT16384Radix4Complex128 tests the size-16384 forward radix-4 kernel (complex128).
func TestForwardDIT16384Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 16384

	skipNaiveReferenceIfSlow(t)

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16384Radix4Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size16384Tol128)
}

// TestInverseDIT16384Radix4Complex128 tests the size-16384 inverse radix-4 kernel (complex128).
func TestInverseDIT16384Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 16384

	skipNaiveReferenceIfSlow(t)

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16384Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex128 failed")
	}

	if !inverseDIT16384Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16384Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size16384Tol128)
}

// TestRoundTripDIT16384Radix4Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT16384Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 16384

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16384Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex128 failed")
	}

	if !inverseDIT16384Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16384Radix4Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size16384Tol128)
}
