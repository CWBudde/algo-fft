//go:build amd64 && asm && !purego

package fft

import (
	"fmt"
	"testing"
)

func TestDebugSize16Radix16_AMD64(t *testing.T) {
	n := 16
	src := make([]complex64, n)
	src[1] = 1

	fwd := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardSSE2Size16Radix16Complex64Asm(fwd, src, twiddle, scratch) {
		t.Fatal("Forward failed")
	}

	fmt.Printf("Inverse Input on AMD64:\n")
	for i, v := range fwd {
		fmt.Printf("[%d] %v\n", i, v)
	}

	// Test Inverse
	roundTrip := make([]complex64, n)
	if !inverseSSE2Size16Radix16Complex64Asm(roundTrip, fwd, twiddle, scratch) {
		t.Fatal("Inverse failed")
	}

	fmt.Printf("Final Output on AMD64:\n")
	for i, v := range fwd {
		fmt.Printf("[%d] %v\n", i, v)
	}
}
