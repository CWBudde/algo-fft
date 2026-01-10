//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestForwardSSE2Size64Radix4Complex128(t *testing.T) {
	const n = 64
	src := randomComplex128(n, 0x12345678)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)
	dst := make([]complex128, n)

	want := make([]complex128, n)
	copy(want, src)
	forwardDIT64Radix4Complex128(want, want, twiddle, scratch)

	if !amd64.ForwardSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size64Radix4Complex128Asm failed")
	}

	assertComplex128Close(t, dst, want, 1e-12)
}

func TestInverseSSE2Size64Radix4Complex128(t *testing.T) {
	const n = 64
	src := randomComplex128(n, 0x87654321)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)
	dst := make([]complex128, n)

	want := make([]complex128, n)
	copy(want, src)
	inverseDIT64Radix4Complex128(want, want, twiddle, scratch)

	if !amd64.InverseSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size64Radix4Complex128Asm failed")
	}

	assertComplex128Close(t, dst, want, 1e-12)
}

func TestRoundTripSSE2Size64Radix4Complex128(t *testing.T) {
	const n = 64
	src := randomComplex128(n, 0xABCDEF)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)

	if !amd64.ForwardSSE2Size64Radix4Complex128Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("Forward failed")
	}
	if !amd64.InverseSSE2Size64Radix4Complex128Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("Inverse failed")
	}

	assertComplex128Close(t, inv, src, 1e-12)
}
