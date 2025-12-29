//go:build fft_asm

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestAVX2Size64Radix4ForwardMatchesReference(t *testing.T) {
	const n = 64
	src := randomComplex64(n, 0x64B00B5)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardAVX2Size64Radix4Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestAVX2Size64Radix4MatchesGoImplementation(t *testing.T) {
	const n = 64
	src := randomComplex64(n, 0x64C0FFEE)

	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size64Radix4Complex64Asm(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
		t.Fatalf("forwardAVX2Size64Radix4Complex64Asm failed")
	}

	dstGo := make([]complex64, n)
	scratchGo := make([]complex64, n)
	if !forwardDIT64Radix4Complex64(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatalf("forwardDIT64Radix4Complex64 failed")
	}

	assertComplex64SliceClose(t, dstAVX2, dstGo, n)
}

func TestAVX2Size64Radix4InverseMatchesReference(t *testing.T) {
	const n = 64
	src := randomComplex64(n, 0x64FEED)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseAVX2Size64Radix4Complex64Asm failed")
	}

	want := reference.NaiveIDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestAVX2Size64Radix4InverseMatchesGoImplementation(t *testing.T) {
	const n = 64
	src := randomComplex64(n, 0x64BAD)

	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseAVX2Size64Radix4Complex64Asm(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
		t.Fatalf("inverseAVX2Size64Radix4Complex64Asm failed")
	}

	dstGo := make([]complex64, n)
	scratchGo := make([]complex64, n)
	if !inverseDIT64Radix4Complex64(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatalf("inverseDIT64Radix4Complex64 failed")
	}

	assertComplex64SliceClose(t, dstAVX2, dstGo, n)
}
