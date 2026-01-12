//go:build amd64 && asm && !purego

package fft

import (
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// getSSE2Kernels returns the SSE2 kernels if available.
func getSSE2Kernels() (forward, inverse Kernel[complex64], available bool) {
	if runtime.GOARCH != "amd64" {
		return nil, nil, false
	}

	features := cpu.DetectFeatures()
	if !features.HasSSE2 {
		return nil, nil, false
	}

	return forwardSSE2Complex64, inverseSSE2Complex64, true
}

func TestSSE2Forward_VsPureGo(t *testing.T) {
	t.Parallel()

	sse2Forward, _, sse2Available := getSSE2Kernels()
	if !sse2Available {
		t.Skip("SSE2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(n))
			twiddle, scratch := prepareFFTData[complex64](n)

			dstGo := make([]complex64, n)
			if !goForward(dstGo, src, twiddle, scratch) {
				t.Fatal("pure-Go forward kernel failed")
			}

			dstSSE2 := make([]complex64, n)
			scratchSSE2 := make([]complex64, n)
			if !sse2Forward(dstSSE2, src, twiddle, scratchSSE2) {
				t.Fatal("SSE2 forward kernel failed")
			}

			const relTol = 1e-5
			if !complexSliceEqual(dstSSE2, dstGo, relTol) {
				t.Errorf("SSE2 forward result differs from pure-Go")
			}
		})
	}
}

func TestSSE2Inverse_VsPureGo(t *testing.T) {
	t.Parallel()

	_, sse2Inverse, sse2Available := getSSE2Kernels()
	if !sse2Available {
		t.Skip("SSE2 not available on this system")
	}

	_, goInverse := getPureGoKernels()

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(n))
			twiddle, scratch := prepareFFTData[complex64](n)

			dstGo := make([]complex64, n)
			if !goInverse(dstGo, src, twiddle, scratch) {
				t.Fatal("pure-Go inverse kernel failed")
			}

			dstSSE2 := make([]complex64, n)
			scratchSSE2 := make([]complex64, n)
			if !sse2Inverse(dstSSE2, src, twiddle, scratchSSE2) {
				t.Fatal("SSE2 inverse kernel failed")
			}

			const relTol = 1e-5
			if !complexSliceEqual(dstSSE2, dstGo, relTol) {
				t.Errorf("SSE2 inverse result differs from pure-Go")
			}
		})
	}
}

// TestSSE2SizeSpecificComplex64 validates size-specific SSE2 kernels against reference implementation.
func TestSSE2SizeSpecificComplex64(t *testing.T) {
	tests := []struct {
		name          string
		size          int
		forward       func([]complex64, []complex64, []complex64, []complex64) bool
		inverse       func([]complex64, []complex64, []complex64, []complex64) bool
		testRoundTrip bool
		testInPlace   bool
	}{
		{
			name:          "Size4_Radix4",
			size:          4,
			forward:       forwardSSE2Size4Radix4Complex64Asm,
			inverse:       inverseSSE2Size4Radix4Complex64Asm,
			testRoundTrip: false,
			testInPlace:   false,
		},
		{
			name:          "Size8_Radix4",
			size:          8,
			forward:       forwardSSE2Size8Radix4Complex64Asm,
			inverse:       inverseSSE2Size8Radix4Complex64Asm,
			testRoundTrip: true,
			testInPlace:   true,
		},
		{
			name:          "Size256_Radix4",
			size:          256,
			forward:       forwardSSE2Size256Radix4Complex64Asm,
			inverse:       inverseSSE2Size256Radix4Complex64Asm,
			testRoundTrip: true,
			testInPlace:   true,
		},
		{
			name:          "Size512_Radix2",
			size:          512,
			forward:       forwardSSE2Size512Radix2Complex64Asm,
			inverse:       inverseSSE2Size512Radix2Complex64Asm,
			testRoundTrip: true,
			testInPlace:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(tc.size, 0xBEEF+uint64(tc.size))
			fwd := make([]complex64, tc.size)
			dst := make([]complex64, tc.size)
			scratch := make([]complex64, tc.size)
			twiddle := ComputeTwiddleFactors[complex64](tc.size)

			// Test correctness vs reference
			if !tc.forward(fwd, src, twiddle, scratch) {
				t.Fatalf("Forward %s failed", tc.name)
			}

			wantFwd := reference.NaiveDFT(src)
			assertComplex64SliceClose(t, fwd, wantFwd, tc.size)

			if !tc.inverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("Inverse %s failed", tc.name)
			}

			wantInv := reference.NaiveIDFT(fwd)
			assertComplex64SliceClose(t, dst, wantInv, tc.size)

			// Test round-trip if enabled
			if tc.testRoundTrip {
				roundtrip := make([]complex64, tc.size)
				if !tc.forward(fwd, src, twiddle, scratch) {
					t.Fatal("Round-trip forward failed")
				}
				if !tc.inverse(roundtrip, fwd, twiddle, scratch) {
					t.Fatal("Round-trip inverse failed")
				}
				assertComplex64SliceClose(t, roundtrip, src, tc.size)
			}

			// Test in-place if enabled
			if tc.testInPlace {
				data := make([]complex64, tc.size)
				copy(data, src)

				if !tc.forward(data, data, twiddle, scratch) {
					t.Fatal("In-place forward failed")
				}
				assertComplex64SliceClose(t, data, wantFwd, tc.size)

				if !tc.inverse(data, data, twiddle, scratch) {
					t.Fatal("In-place inverse failed")
				}
				assertComplex64SliceClose(t, data, src, tc.size)
			}
		})
	}
}

// TestSSE2SizeSpecificComplex128 validates size-specific SSE2 kernels for complex128.
func TestSSE2SizeSpecificComplex128(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		forward func([]complex128, []complex128, []complex128, []complex128) bool
		inverse func([]complex128, []complex128, []complex128, []complex128) bool
	}{
		{
			name:    "Size4_Radix4",
			size:    4,
			forward: forwardSSE2Size4Radix4Complex128Asm,
			inverse: inverseSSE2Size4Radix4Complex128Asm,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(tc.size, 0xFEED+uint64(tc.size))
			fwd := make([]complex128, tc.size)
			dst := make([]complex128, tc.size)
			scratch := make([]complex128, tc.size)
			twiddle := ComputeTwiddleFactors[complex128](tc.size)

			if !tc.forward(fwd, src, twiddle, scratch) {
				t.Fatalf("Forward %s failed", tc.name)
			}

			wantFwd := reference.NaiveDFT128(src)
			assertComplex128SliceClose(t, fwd, wantFwd, tc.size)

			if !tc.inverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("Inverse %s failed", tc.name)
			}

			wantInv := reference.NaiveIDFT128(fwd)
			assertComplex128SliceClose(t, dst, wantInv, tc.size)
		})
	}
}
