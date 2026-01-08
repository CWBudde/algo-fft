//go:build amd64 && asm && !purego

package kernels

import (
	"fmt"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func TestTraceAVX2(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	// Use simple input for easier debugging
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), float32(i)*0.1)
	}
	
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	// Run AVX2 version
	result := amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch, bitrev)
	fmt.Printf("AVX2 kernel returned: %v\n", result)
	
	// Print first few output values
	fmt.Println("First 10 dst values:")
	for i := 0; i < 10; i++ {
		fmt.Printf("  dst[%d] = %v\n", i, dst[i])
	}
	
	fmt.Println("\nDst values at k1=1:")
	for k2 := 0; k2 < 5; k2++ {
		i := 1*32 + k2
		fmt.Printf("  dst[%d] = %v\n", i, dst[i])
	}
}
