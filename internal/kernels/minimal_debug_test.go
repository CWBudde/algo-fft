//go:build amd64 && asm && !purego

package kernels

import (
	"fmt"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func TestMinimalDebug(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	// Create a simple impulse input: all zeros except first element = 1
	src := make([]complex64, n)
	src[0] = 1.0
	
	dstGo := make([]complex64, n)
	dstAVX2 := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	// Run both versions
	if !forwardDIT512Mixed16x32Complex64(dstGo, src, twiddle, scratch, bitrev) {
		t.Fatal("Go failed")
	}
	
	if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dstAVX2, src, twiddle, scratch, bitrev) {
		t.Fatal("AVX2 failed")
	}

	// For an impulse input, FFT output should be all 1s (or close to it with twiddle scaling)
	fmt.Println("Impulse test - first 20 outputs:")
	fmt.Println("Go:")
	for i := 0; i < 20; i++ {
		fmt.Printf("  [%2d] = %v\n", i, dstGo[i])
	}
	fmt.Println("\nAVX2:")
	for i := 0; i < 20; i++ {
		fmt.Printf("  [%2d] = %v\n", i, dstAVX2[i])
	}
}
