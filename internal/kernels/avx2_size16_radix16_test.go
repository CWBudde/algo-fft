//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardAVX2Size16Radix16Complex64 tests the AVX2 size-16 radix-16 forward kernel
func TestForwardAVX2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0x11223344)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !amd64.ForwardAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size16Radix16Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestInverseAVX2Size16Radix16Complex64 tests the AVX2 size-16 radix-16 inverse kernel
func TestInverseAVX2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0x55667788)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	// Use reference to get valid frequency domain input
	fwdRef := reference.NaiveDFT(src)
	for i := range fwdRef {
		fwd[i] = complex64(fwdRef[i])
	}

	if !amd64.InverseAVX2Size16Radix16Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size16Radix16Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwdRef)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestRoundTripAVX2Size16Radix16Complex64 tests forward-inverse round-trip
func TestRoundTripAVX2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0x99AABBCC)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !amd64.ForwardAVX2Size16Radix16Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forward failed")
	}

	if !amd64.InverseAVX2Size16Radix16Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverse failed")
	}

	assertComplex64Close(t, inv, src, 1e-6)
}