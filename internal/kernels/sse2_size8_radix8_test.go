//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardSSE2Size8Radix8Complex64 tests the SSE2 size-8 radix-8 forward kernel
func TestForwardSSE2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	// Radix-8 kernel expects natural order input
	bitrev := math.ComputeIdentityIndices(n)

	if !amd64.ForwardSSE2Size8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size8Radix8Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestInverseSSE2Size8Radix8Complex64 tests the SSE2 size-8 radix-8 inverse kernel
func TestInverseSSE2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	// Radix-8 kernel expects natural order input
	bitrev := math.ComputeIdentityIndices(n)

	// Use reference forward to ensure valid input
	fwdRef := reference.NaiveDFT(src)
	for i := range fwdRef {
		fwd[i] = complex64(fwdRef[i])
	}

	if !amd64.InverseSSE2Size8Radix8Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size8Radix8Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwdRef)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestRoundTripSSE2Size8Radix8Complex64 tests forward-inverse round-trip
func TestRoundTripSSE2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xBEEFCAFE)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := math.ComputeIdentityIndices(n)

	// Forward transform
	if !amd64.ForwardSSE2Size8Radix8Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forward transform failed")
	}

	// Inverse transform
	if !amd64.InverseSSE2Size8Radix8Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverse transform failed")
	}

	// Verify round-trip: inv should equal src
	assertComplex64Close(t, inv, src, 1e-6)
}
