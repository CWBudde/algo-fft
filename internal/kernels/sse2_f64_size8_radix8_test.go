//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestForwardSSE2Size8Radix8Complex128(t *testing.T) {
	const n = 8
	src := randomComplex128(n, 0x12345678)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	dst := make([]complex128, n)

	want := make([]complex128, n)
	copy(want, src)
	forwardDIT8Radix8Complex128(want, want, twiddle, scratch)

	if !amd64.ForwardSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size8Radix8Complex128Asm failed")
	}

	assertComplex128Close(t, dst, want, 1e-12)
}

func TestInverseSSE2Size8Radix8Complex128(t *testing.T) {
	const n = 8
	src := randomComplex128(n, 0x87654321)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	dst := make([]complex128, n)

	want := make([]complex128, n)
	copy(want, src)
	inverseDIT8Radix8Complex128(want, want, twiddle, scratch)

	if !amd64.InverseSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size8Radix8Complex128Asm failed")
	}

	assertComplex128Close(t, dst, want, 1e-12)
}

func TestRoundTripSSE2Size8Radix8Complex128(t *testing.T) {
	const n = 8
	src := randomComplex128(n, 0xABCDEF)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)

	if !amd64.ForwardSSE2Size8Radix8Complex128Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("Forward failed")
	}
	if !amd64.InverseSSE2Size8Radix8Complex128Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("Inverse failed")
	}

	assertComplex128Close(t, inv, src, 1e-12)
}
