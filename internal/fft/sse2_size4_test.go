//go:build amd64 && fft_asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestSSE2Size4Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4
	src := randomComplex128(n, 0xBEEF)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardSSE2Size4Radix4Complex128Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardSSE2Size4Radix4Complex128Asm failed")
	}

	if !inverseSSE2Size4Radix4Complex128Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseSSE2Size4Radix4Complex128Asm failed")
	}

	wantFwd := reference.NaiveDFT128(src)
	assertComplex128SliceClose(t, fwd, wantFwd, n)

	wantInv := reference.NaiveIDFT128(fwd)
	assertComplex128SliceClose(t, dst, wantInv, n)
}
