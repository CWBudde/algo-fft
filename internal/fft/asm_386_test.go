//go:build 386 && asm && !purego

package fft

import (
	"fmt"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestSSE2SizeSpecificComplex64_386(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		forward func([]complex64, []complex64, []complex64, []complex64, []int) bool
		inverse func([]complex64, []complex64, []complex64, []complex64, []int) bool
		radix   int // 2 or 4 to choose correct bit reversal
	}{
		{
			name:    "Size2_Radix2",
			size:    2,
			forward: forwardSSE2Size2Radix2Complex64Asm,
			inverse: inverseSSE2Size2Radix2Complex64Asm,
			radix:   2,
		},
		{
			name:    "Size4_Radix4",
			size:    4,
			forward: forwardSSE2Size4Radix4Complex64Asm,
			inverse: inverseSSE2Size4Radix4Complex64Asm,
			radix:   4,
		},
		{
			name:    "Size8_Radix2",
			size:    8,
			forward: forwardSSE2Size8Radix2Complex64Asm,
			inverse: inverseSSE2Size8Radix2Complex64Asm,
			radix:   2,
		},
		{
			name:    "Size16_Radix4",
			size:    16,
			forward: forwardSSE2Size16Radix4Complex64Asm,
			inverse: inverseSSE2Size16Radix4Complex64Asm,
			radix:   4,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := randomComplex64(tc.size, 0xBEEF)
			fwd := make([]complex64, tc.size)
			dst := make([]complex64, tc.size)
			scratch := make([]complex64, tc.size)
			twiddle := ComputeTwiddleFactors[complex64](tc.size)
			
			var bitrev []int
			if tc.radix == 4 {
				bitrev = math.ComputeBitReversalIndicesRadix4(tc.size)
			} else {
				bitrev = ComputeBitReversalIndices(tc.size)
			}

			if !tc.forward(fwd, src, twiddle, scratch, bitrev) {
				t.Fatalf("Forward %s failed", tc.name)
			}

			if !tc.inverse(dst, fwd, twiddle, scratch, bitrev) {
				t.Fatalf("Inverse %s failed", tc.name)
			}

			wantFwd := reference.NaiveDFT(src)
			assertComplex64SliceClose(t, fwd, wantFwd, tc.size)

			wantInv := reference.NaiveIDFT(fwd)
			assertComplex64SliceClose(t, dst, wantInv, tc.size)
		})
	}
}

func TestSSE2SizeSpecificComplex128_386(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		forward func([]complex128, []complex128, []complex128, []complex128, []int) bool
		inverse func([]complex128, []complex128, []complex128, []complex128, []int) bool
		radix   int
	}{
		{
			name:    "Size2_Radix2",
			size:    2,
			forward: forwardSSE2Size2Radix2Complex128Asm,
			inverse: inverseSSE2Size2Radix2Complex128Asm,
			radix:   2,
		},
		{
			name:    "Size4_Radix4",
			size:    4,
			forward: forwardSSE2Size4Radix4Complex128Asm,
			inverse: inverseSSE2Size4Radix4Complex128Asm,
			radix:   4,
		},
		{
			name:    "Size8_Radix2",
			size:    8,
			forward: forwardSSE2Size8Radix2Complex128Asm,
			inverse: inverseSSE2Size8Radix2Complex128Asm,
			radix:   2,
		},
		{
			name:    "Size16_Radix4",
			size:    16,
			forward: forwardSSE2Size16Radix4Complex128Asm,
			inverse: inverseSSE2Size16Radix4Complex128Asm,
			radix:   4,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := randomComplex128(tc.size, 0xFEED)
			fwd := make([]complex128, tc.size)
			dst := make([]complex128, tc.size)
			scratch := make([]complex128, tc.size)
			twiddle := ComputeTwiddleFactors[complex128](tc.size)
			
			var bitrev []int
			if tc.radix == 4 {
				bitrev = math.ComputeBitReversalIndicesRadix4(tc.size)
			} else {
				bitrev = ComputeBitReversalIndices(tc.size)
			}

			if !tc.forward(fwd, src, twiddle, scratch, bitrev) {
				t.Fatalf("Forward %s failed", tc.name)
			}

			if !tc.inverse(dst, fwd, twiddle, scratch, bitrev) {
				t.Fatalf("Inverse %s failed", tc.name)
			}

			wantFwd := reference.NaiveDFT128(src)
			assertComplex128SliceClose(t, fwd, wantFwd, tc.size)

			wantInv := reference.NaiveIDFT128(fwd)
			assertComplex128SliceClose(t, dst, wantInv, tc.size)
		})
	}
}

// TestSSE2Forward_VsPureGo_386 validates SSE2 assembly implementations
// against pure-Go DIT kernels across various FFT sizes.
func TestSSE2Forward_VsPureGo_386(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	if !features.HasSSE2 {
		t.Skip("SSE2 not available on this system")
	}

	sse2Forward := forwardSSE2Complex64
	goForward := forwardDITComplex64

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, uint64(n))
			twiddle := ComputeTwiddleFactors[complex64](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex64, n)

			dstGo := make([]complex64, n)
			if !goForward(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("pure-Go forward kernel failed")
			}

			dstSSE2 := make([]complex64, n)
			scratchSSE2 := make([]complex64, n)
			if !sse2Forward(dstSSE2, src, twiddle, scratchSSE2, bitrev) {
				t.Fatal("SSE2 forward kernel failed")
			}

			assertComplex64SliceClose(t, dstSSE2, dstGo, n)
		})
	}
}

// TestSSE2Inverse_VsPureGo_386 validates SSE2 inverse FFT implementations
// against pure-Go DIT kernels across various FFT sizes.
func TestSSE2Inverse_VsPureGo_386(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	if !features.HasSSE2 {
		t.Skip("SSE2 not available on this system")
	}

	sse2Inverse := inverseSSE2Complex64
	goInverse := inverseDITComplex64

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, uint64(n))
			twiddle := ComputeTwiddleFactors[complex64](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex64, n)

			dstGo := make([]complex64, n)
			if !goInverse(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("pure-Go inverse kernel failed")
			}

			dstSSE2 := make([]complex64, n)
			scratchSSE2 := make([]complex64, n)
			if !sse2Inverse(dstSSE2, src, twiddle, scratchSSE2, bitrev) {
				t.Fatal("SSE2 inverse kernel failed")
			}

			assertComplex64SliceClose(t, dstSSE2, dstGo, n)
		})
	}
}
