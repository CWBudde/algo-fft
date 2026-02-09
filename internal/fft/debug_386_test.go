//go:build 386 && asm && !purego && debug

package fft

import (
	"fmt"
	"testing"

	"github.com/cwbudde/algo-fft/internal/math"
)

func TestDebugSize16Radix16_386(t *testing.T) {
	n := 16
	src := make([]complex64, n)
	src[1] = 1

	fwd := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := math.ComputeIdentityIndices(n)

	if !forwardSSE3Size16Radix16Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("Forward failed")
	}

	fmt.Printf("Forward Output on 386:\n")
	for i, v := range fwd {
		fmt.Printf("[%d] %v\n", i, v)
	}

	roundTrip := make([]complex64, n)
	if !inverseSSE3Size16Radix16Complex64Asm(roundTrip, fwd, twiddle, scratch, bitrev) {
		t.Fatal("Inverse failed")
	}

	fmt.Printf("Round-trip Output on 386:\n")
	for i, v := range roundTrip {
		fmt.Printf("[%d] %v\n", i, v)
	}
}
