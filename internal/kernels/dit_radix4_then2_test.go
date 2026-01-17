package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	radix4Then2Tol64 = 1e-4
)

// TestForwardRadix4Then2Complex64 tests the radix-4-then-2 forward kernel.
func TestForwardRadix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardRadix4Then2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardRadix4Then2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, radix4Then2Tol64)
}

// TestInverseRadix4Then2Complex64 tests the radix-4-then-2 inverse kernel.
func TestInverseRadix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardRadix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardRadix4Then2Complex64 failed")
	}

	if !inverseRadix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseRadix4Then2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, radix4Then2Tol64)
}
