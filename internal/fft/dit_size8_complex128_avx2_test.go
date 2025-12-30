//go:build amd64 && fft_asm && !purego

package fft

import (
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestAVX2Size8Radix2Complex128(t *testing.T) {
	requireAVX2(t)

	n := 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(i+1)*0.5)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	// Forward
	if !forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardAVX2Size8Radix2Complex128Asm failed")
	}

	want := reference.NaiveDFT128(src)
	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > 1e-12 {
			t.Errorf("Forward[%d] = %v, want %v", i, dst[i], want[i])
		}
	}

	// Inverse
	srcInv := make([]complex128, n)
	copy(srcInv, dst)
	if !inverseAVX2Size8Radix2Complex128Asm(dst, srcInv, twiddle, scratch, bitrev) {
		t.Fatal("inverseAVX2Size8Radix2Complex128Asm failed")
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-src[i]) > 1e-12 {
			t.Errorf("Inverse[%d] = %v, want %v", i, dst[i], src[i])
		}
	}
}
