//go:build 386 && asm && !purego

package fft

import (
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
			} else if tc.radix == 0 {
				bitrev = math.ComputeIdentityIndices(tc.size)
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

func TestSelectKernelsComplex64_SSEOnly_386(t *testing.T) {
	cpu.ForceSSEOnlyForTests()
	defer cpu.ResetDetection()

	features := cpu.DetectFeatures()
	if !features.HasSSE || features.HasSSE2 {
		t.Fatalf("expected SSE-only features, got: %+v", features)
	}

	kernels := selectKernelsComplex64(features)
	if kernels.Forward == nil || kernels.Inverse == nil {
		t.Fatal("expected SSE-only kernels to be available")
	}

	const n = 32
	src := randomComplex64(n, 0x4EED)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := math.ComputeBitReversalIndices(n)

	if !kernels.Forward(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("SSE-only forward kernel failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
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
