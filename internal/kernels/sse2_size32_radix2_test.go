//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestForwardSSE2Size32Radix2Complex64(t *testing.T) {
	const n = 32
	src := randomComplex64(n, 0x12345678)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	want := make([]complex64, n)
	copy(want, src)
	forwardDIT32Complex64(want, want, twiddle, scratch)

	if !amd64.ForwardSSE2Size32Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size32Radix2Complex64Asm failed")
	}

	assertComplex64Close(t, dst, want, 1e-4)
}

func TestInverseSSE2Size32Radix2Complex64(t *testing.T) {
	const n = 32
	src := randomComplex64(n, 0x87654321)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	want := make([]complex64, n)
	copy(want, src)
	inverseDIT32Complex64(want, want, twiddle, scratch)

	if !amd64.InverseSSE2Size32Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size32Radix2Complex64Asm failed")
	}

	assertComplex64Close(t, dst, want, 1e-4)
}

func TestRoundTripSSE2Size32Radix2Complex64(t *testing.T) {
	const n = 32
	src := randomComplex64(n, 0xABCDEF)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)

	if !amd64.ForwardSSE2Size32Radix2Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("Forward failed")
	}
	if !amd64.InverseSSE2Size32Radix2Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("Inverse failed")
	}

	assertComplex64Close(t, inv, src, 1e-4)
}
