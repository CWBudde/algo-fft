//go:build 386 && asm && !purego

package fft

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func generateRandomComplex64_386(n int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF))
	result := make([]complex64, n)

	for i := range result {
		re := rng.Float32()*2 - 1
		im := rng.Float32()*2 - 1
		result[i] = complex(re, im)
	}

	return result
}

func complexSliceEqual_386(a, b []complex64, relTol float32) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if !complexNearEqual_386(a[i], b[i], relTol) {
			return false
		}
	}

	return true
}

func complexNearEqual_386(a, b complex64, relTol float32) bool {
	diff := a - b
	diffMag := float32(real(diff)*real(diff) + imag(diff)*imag(diff))
	bMag := float32(real(b)*real(b) + imag(b)*imag(b))

	if bMag > 1e-10 {
		return diffMag <= relTol*relTol*bMag
	}

	return diffMag <= relTol*relTol
}

func prepareFFTData_386(n int) ([]complex64, []int, []complex64) {
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)

	return twiddle, bitrev, scratch
}

func sizeString_386(n int) string {
	return fmt.Sprintf("N=%d", n)
}

func getSSE2Kernels_386() (forward, inverse Kernel[complex64], available bool) {
	features := cpu.DetectFeatures()
	if !features.HasSSE2 {
		return nil, nil, false
	}

	return forwardSSE2Complex64, inverseSSE2Complex64, true
}

func TestSSE2Forward_VsPureGo_386(t *testing.T) {
	t.Parallel()

	sse2Forward, _, sse2Available := getSSE2Kernels_386()
	if !sse2Available {
		t.Skip("SSE2 not available on this system")
	}

	goForward := forwardDITComplex64

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(sizeString_386(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64_386(n, uint64(n))
			twiddle, bitrev, scratch := prepareFFTData_386(n)

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
			if !complexSliceEqual_386(dstSSE2, dstGo, relTol) {
				t.Errorf("SSE2 forward result differs from pure-Go")
			}
		})
	}
}

func TestSSE2Inverse_VsPureGo_386(t *testing.T) {
	t.Parallel()

	_, sse2Inverse, sse2Available := getSSE2Kernels_386()
	if !sse2Available {
		t.Skip("SSE2 not available on this system")
	}

	goInverse := inverseDITComplex64

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(sizeString_386(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64_386(n, uint64(n))
			twiddle, bitrev, scratch := prepareFFTData_386(n)

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
			if !complexSliceEqual_386(dstSSE2, dstGo, relTol) {
				t.Errorf("SSE2 inverse result differs from pure-Go")
			}
		})
	}
}
