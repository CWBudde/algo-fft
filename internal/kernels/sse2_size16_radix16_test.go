//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardSSE2Size16Radix16Complex64 tests the SSE2 size-16 radix-16 forward kernel
func TestForwardSSE2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xABCDEFFF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !amd64.ForwardSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch) {
		t.Fatal("ForwardSSE2Size16Radix16Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestInverseSSE2Size16Radix16Complex64 tests the SSE2 size-16 radix-16 inverse kernel
func TestInverseSSE2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0x12345678)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	fwdRef := reference.NaiveDFT(src)
	for i := range fwdRef {
		fwd[i] = complex64(fwdRef[i])
	}

	if !amd64.InverseSSE2Size16Radix16Complex64Asm(dst, fwd, twiddle, scratch) {
		t.Fatal("InverseSSE2Size16Radix16Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwdRef)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestRoundTripSSE2Size16Radix16Complex64 tests forward-inverse round-trip
func TestRoundTripSSE2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xDEADBEEF)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !amd64.ForwardSSE2Size16Radix16Complex64Asm(fwd, src, twiddle, scratch) {
		t.Fatal("forward failed")
	}

	if !amd64.InverseSSE2Size16Radix16Complex64Asm(inv, fwd, twiddle, scratch) {
		t.Fatal("inverse failed")
	}

	assertComplex64Close(t, inv, src, 1e-6)
}
