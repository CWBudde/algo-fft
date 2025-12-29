//go:build fft_asm

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestAVX2Size16Radix4ForwardMatchesReference(t *testing.T) {
	const n = 16
	src := randomComplex64(n, 0x16B00B5)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardAVX2Size16Radix4Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestAVX2Size16Radix4InverseMatchesReference(t *testing.T) {
	const n = 16
	src := randomComplex64(n, 0x1D1F7)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseAVX2Size16Radix4Complex64Asm failed")
	}

	want := reference.NaiveIDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestAVX2Size16Radix4MatchesGoImplementation(t *testing.T) {
	const n = 16
	src := randomComplex64(n, 0xC16E5A)

	// AVX2 implementation
	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size16Radix4Complex64Asm(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
		t.Fatalf("forwardAVX2Size16Radix4Complex64Asm failed")
	}

	// Go optimized implementation
	dstGo := make([]complex64, n)
	scratchGo := make([]complex64, n)

	if !forwardDIT16Radix4Complex64(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatalf("forwardDIT16Radix4Complex64 failed")
	}

	assertComplex64SliceClose(t, dstAVX2, dstGo, n)
}

func TestAVX2Size16Radix4InverseMatchesGoImplementation(t *testing.T) {
	const n = 16
	src := randomComplex64(n, 0xBADF00D)

	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseAVX2Size16Radix4Complex64Asm(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
		t.Fatalf("inverseAVX2Size16Radix4Complex64Asm failed")
	}

	dstGo := make([]complex64, n)
	scratchGo := make([]complex64, n)
	if !inverseDIT16Radix4Complex64(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatalf("inverseDIT16Radix4Complex64 failed")
	}

	assertComplex64SliceClose(t, dstAVX2, dstGo, n)
}
