//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardSSE2Size64Radix4Complex64 tests the SSE2 size-64 radix-4 forward kernel
func TestForwardSSE2Size64Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 64
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !amd64.ForwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size64Radix4Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size64Tol64)
}

// TestInverseSSE2Size64Radix4Complex64 tests the SSE2 size-64 radix-4 inverse kernel
func TestInverseSSE2Size64Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 64
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !amd64.ForwardSSE2Size64Radix4Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size64Radix4Complex64Asm failed")
	}

	if !amd64.InverseSSE2Size64Radix4Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSE2Size64Radix4Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size64Tol64)
}

// TestRoundTripSSE2Size64Radix4Complex64 tests forward-inverse round-trip
func TestRoundTripSSE2Size64Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 64
	src := randomComplex64(n, 0xBEEFCAFE)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Forward transform
	if !amd64.ForwardSSE2Size64Radix4Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forward transform failed")
	}

	// Inverse transform
	if !amd64.InverseSSE2Size64Radix4Complex64Asm(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverse transform failed")
	}

	// Verify round-trip: inv should equal src
	assertComplex64Close(t, inv, src, size64Tol64)
}
