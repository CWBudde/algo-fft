//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardAVX2Size8Radix8Complex64 tests the AVX2 size-8 radix-8 forward kernel
func TestForwardAVX2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0x12345678)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	// Radix-8 kernel expects natural order input
	bitrev := math.ComputeIdentityIndices(n)

	if !amd64.ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size8Radix8Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestInverseAVX2Size8Radix8Complex64 tests the AVX2 size-8 radix-8 inverse kernel
func TestInverseAVX2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0x87654321)
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

	if !amd64.InverseAVX2Size8Radix8Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size8Radix8Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwdRef)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestRoundTripAVX2Size8Radix8Complex64 tests forward-inverse round-trip
func TestRoundTripAVX2Size8Radix8Complex64(t *testing.T) {
	t.Parallel()

	const n = 8
	src := randomComplex64(n, 0xAABBCCDD)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := math.ComputeIdentityIndices(n)

	if !amd64.ForwardAVX2Size8Radix8Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size8Radix8Complex64Asm failed")
	}

	if !amd64.InverseAVX2Size8Radix8Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size8Radix8Complex64Asm failed")
	}

	assertComplex64Close(t, inv, src, 1e-6)
}
