//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardAVX2Size4Radix4Complex64 tests the AVX2 size-4 radix-4 forward kernel
func TestForwardAVX2Size4Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4
	src := randomComplex64(n, 0x12345678)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := make([]complex64, n) // Not used for size 4
	bitrev := make([]int, n)        // Not used for size 4

	if !amd64.ForwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size4Radix4Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestInverseAVX2Size4Radix4Complex64 tests the AVX2 size-4 radix-4 inverse kernel
func TestInverseAVX2Size4Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4
	src := randomComplex64(n, 0x87654321)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := make([]complex64, n) // Not used for size 4
	bitrev := make([]int, n)        // Not used for size 4

	// Generate forward data using reference to ensure valid input
	fwdRef := reference.NaiveDFT(src)
	for i := range fwdRef {
		fwd[i] = complex64(fwdRef[i])
	}

	if !amd64.InverseAVX2Size4Radix4Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size4Radix4Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwdRef)
	assertComplex64Close(t, dst, want, 1e-6)
}

// TestRoundTripAVX2Size4Radix4Complex64 tests forward-inverse round-trip
func TestRoundTripAVX2Size4Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4
	src := randomComplex64(n, 0xAABBCCDD)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := make([]complex64, n)
	bitrev := make([]int, n)

	if !amd64.ForwardAVX2Size4Radix4Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size4Radix4Complex64Asm failed")
	}

	if !amd64.InverseAVX2Size4Radix4Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size4Radix4Complex64Asm failed")
	}

	assertComplex64Close(t, inv, src, 1e-6)
}
