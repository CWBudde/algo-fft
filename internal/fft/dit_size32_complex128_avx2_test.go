//go:build amd64 && fft_asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestAVX2Size32Complex128ForwardMatchesReference(t *testing.T) {
	requireAVX2(t)

	const n = 32
	src := randomComplex128(n, 0x32C128)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardAVX2Size32Complex128Asm failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

func TestAVX2Size32Complex128InverseMatchesReference(t *testing.T) {
	requireAVX2(t)

	const n = 32
	src := randomComplex128(n, 0x321128)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseAVX2Size32Complex128Asm failed")
	}

	want := reference.NaiveIDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}
