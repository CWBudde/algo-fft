//go:build amd64 && asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestSSE2Size256Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex64(n, 0xDEAD256)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardSSE2Size256Radix4Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardSSE2Size256Radix4Complex64Asm failed")
	}

	wantFwd := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, fwd, wantFwd, n)

	if !inverseSSE2Size256Radix4Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseSSE2Size256Radix4Complex64Asm failed")
	}

	wantInv := reference.NaiveIDFT(fwd)
	assertComplex64SliceClose(t, dst, wantInv, n)
}

func TestSSE2Size256Radix4Complex64_RoundTrip(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex64(n, 0xCAFE256)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardSSE2Size256Radix4Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardSSE2Size256Radix4Complex64Asm failed")
	}

	if !inverseSSE2Size256Radix4Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseSSE2Size256Radix4Complex64Asm failed")
	}

	// Round-trip should recover original input
	assertComplex64SliceClose(t, dst, src, n)
}

func TestSSE2Size256Radix4Complex64_InPlace(t *testing.T) {
	t.Parallel()

	const n = 256
	src := randomComplex64(n, 0xBEEF256)
	data := make([]complex64, n)
	copy(data, src)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// In-place forward
	if !forwardSSE2Size256Radix4Complex64Asm(data, data, twiddle, scratch, bitrev) {
		t.Fatal("in-place forward failed")
	}

	wantFwd := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, data, wantFwd, n)

	// In-place inverse
	if !inverseSSE2Size256Radix4Complex64Asm(data, data, twiddle, scratch, bitrev) {
		t.Fatal("in-place inverse failed")
	}

	// Should recover original
	assertComplex64SliceClose(t, data, src, n)
}
