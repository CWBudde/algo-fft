package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	mixedRadix24Tol64 = 1e-4
)

// TestForwardMixedRadix24Complex64 tests the mixed-radix-2/4 forward kernel
func TestForwardMixedRadix24Complex64(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardMixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardMixedRadix24Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, mixedRadix24Tol64)
}

// TestInverseMixedRadix24Complex64 tests the mixed-radix-2/4 inverse kernel
func TestInverseMixedRadix24Complex64(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardMixedRadix24Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardMixedRadix24Complex64 failed")
	}

	if !inverseMixedRadix24Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseMixedRadix24Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, mixedRadix24Tol64)
}
