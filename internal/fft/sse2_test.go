//go:build amd64 && asm && !purego

package fft

import (
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
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
			twiddle, bitrev, scratch := prepareFFTData(n)

			dstGo := make([]complex64, n)
			if !goForward(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("pure-Go forward kernel failed")
			}

			dstSSE2 := make([]complex64, n)
			scratchSSE2 := make([]complex64, n)
			if !sse2Forward(dstSSE2, src, twiddle, scratchSSE2, bitrev) {
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
			twiddle, bitrev, scratch := prepareFFTData(n)

			dstGo := make([]complex64, n)
			if !goInverse(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("pure-Go inverse kernel failed")
			}

			dstSSE2 := make([]complex64, n)
			scratchSSE2 := make([]complex64, n)
			if !sse2Inverse(dstSSE2, src, twiddle, scratchSSE2, bitrev) {
				t.Fatal("SSE2 inverse kernel failed")
			}

			const relTol = 1e-5
			if !complexSliceEqual(dstSSE2, dstGo, relTol) {
				t.Errorf("SSE2 inverse result differs from pure-Go")
			}
		})
	}
}
