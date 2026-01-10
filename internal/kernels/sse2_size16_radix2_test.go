//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardSSE2Size16Radix2Complex64 tests the SSE2 size-16 radix-2 forward kernel
func TestForwardSSE2Size16Radix2Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !amd64.ForwardSSE2Size16Radix2Complex64Asm(dst, src, twiddle, scratch) {
		t.Fatal("ForwardSSE2Size16Radix2Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-5) // Relax tolerance slightly for larger sizes if needed, but 1e-5 should hold
}

// TestInverseSSE2Size16Radix2Complex64 tests the SSE2 size-16 radix-2 inverse kernel
func TestInverseSSE2Size16Radix2Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	// Use reference forward to test inverse in isolation if possible,
	// but using assembly forward is fine if verified above.
	// Let's use naive for 'fwd' to ensure inverse is tested against correct input
	fwdRef := reference.NaiveDFT(src)
	for i := range fwdRef {
		fwd[i] = complex64(fwdRef[i])
	}

	if !amd64.InverseSSE2Size16Radix2Complex64Asm(dst, fwd, twiddle, scratch) {
		t.Fatal("InverseSSE2Size16Radix2Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwdRef)
	assertComplex64Close(t, dst, want, 1e-5)
}

// TestRoundTripSSE2Size16Radix2Complex64 tests forward-inverse round-trip
func TestRoundTripSSE2Size16Radix2Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xBEEFCAFE)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	// Forward transform
	if !amd64.ForwardSSE2Size16Radix2Complex64Asm(fwd, src, twiddle, scratch) {
		t.Fatal("forward transform failed")
	}

	// Inverse transform
	if !amd64.InverseSSE2Size16Radix2Complex64Asm(inv, fwd, twiddle, scratch) {
		t.Fatal("inverse transform failed")
	}

	// Verify round-trip: inv should equal src
	assertComplex64Close(t, inv, src, 1e-5)
}
