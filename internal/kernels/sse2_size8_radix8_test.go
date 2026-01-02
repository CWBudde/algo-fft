//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardSSE2Size8Radix8Complex64 tests the SSE2 size-8 radix-8 forward kernel
func TestForwardSSE2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xFEEDFACE)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n) // Standard bit reversal

	// Kernel expects bit-reversed input
	srcPerm := make([]complex64, n)
	for i, j := range bitrev {
		srcPerm[i] = src[j]
	}

	if !amd64.ForwardSSE2Size8Radix8Complex64Asm(dst, srcPerm, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size8Radix8Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestInverseSSE2Size8Radix8Complex64 tests the SSE2 size-8 radix-8 inverse kernel
func TestInverseSSE2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xDECAFBAD)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n) // Standard bit reversal

	// Permute src for Forward
	srcPerm := make([]complex64, n)
	for i, j := range bitrev {
		srcPerm[i] = src[j]
	}

	if !amd64.ForwardSSE2Size8Radix8Complex64Asm(fwd, srcPerm, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size8Radix8Complex64Asm failed")
	}

	// Permute fwd for Inverse
	fwdPerm := make([]complex64, n)
	for i, j := range bitrev {
		fwdPerm[i] = fwd[j]
	}

	if !amd64.InverseSSE2Size8Radix8Complex64Asm(dst, fwdPerm, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size8Radix8Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestRoundTripSSE2Size8Radix8Complex64 tests forward-inverse round-trip
func TestRoundTripSSE2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xBADCAFFE)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n) // Standard bit reversal

	// Permute src for Forward
	srcPerm := make([]complex64, n)
	for i, j := range bitrev {
		srcPerm[i] = src[j]
	}

	// Forward transform
	if !amd64.ForwardSSE2Size8Radix8Complex64Asm(fwd, srcPerm, twiddle, scratch, bitrev) {
		t.Fatal("forward transform failed")
	}

	// Permute fwd for Inverse
	fwdPerm := make([]complex64, n)
	for i, j := range bitrev {
		fwdPerm[i] = fwd[j]
	}

	// Inverse transform
	if !amd64.InverseSSE2Size8Radix8Complex64Asm(inv, fwdPerm, twiddle, scratch, bitrev) {
		t.Fatal("inverse transform failed")
	}

	// Verify round-trip: inv should equal src
	assertComplex64Close(t, inv, src, 1e-6)
}
