//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardSSE2Size128Radix4Complex64(t *testing.T) {
	const n = 128
	src := randomComplex64(n, 0x1234ABCD)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)
	dst := make([]complex64, n)

	if !amd64.ForwardSSE2Size128Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size128Radix4Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestInverseSSE2Size128Radix4Complex64(t *testing.T) {
	const n = 128
	src := randomComplex64(n, 0xDEADBEEF)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)
	dst := make([]complex64, n)

	if !amd64.InverseSSE2Size128Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size128Radix4Complex64Asm failed")
	}

	want := reference.NaiveIDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestRoundTripSSE2Size128Radix4Complex64(t *testing.T) {
	const n = 128
	src := randomComplex64(n, 0xBEEF1234)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)

	if !amd64.ForwardSSE2Size128Radix4Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("Forward failed")
	}
	if !amd64.InverseSSE2Size128Radix4Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("Inverse failed")
	}

	assertComplex64SliceClose(t, inv, src, n)
}
