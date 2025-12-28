//go:build fft_asm

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestAVX2Size256Radix4ForwardMatchesReference(t *testing.T) {
	const n = 256
	src := randomComplex64(n, 0xBAD214)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardAVX2Size256Radix4Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestAVX2Size256Radix4MatchesGoImplementation(t *testing.T) {
	const n = 256
	src := randomComplex64(n, 0xC0FFEE)

	// AVX2 implementation
	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size256Radix4Complex64Asm(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
		t.Fatalf("forwardAVX2Size256Radix4Complex64Asm failed")
	}

	// Go optimized implementation
	dstGo := make([]complex64, n)
	scratchGo := make([]complex64, n)

	if !forwardDIT256Radix4Complex64(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	// Both should produce identical results
	assertComplex64SliceClose(t, dstAVX2, dstGo, n)
}
