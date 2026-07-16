//go:build amd64 && !purego

package fft

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// TestRecombineForwardSSE3MatchesGeneric verifies the SSE3 tier of the
// forward recombination against the scalar loop. AVX2 hosts never take this
// tier in normal runs, so it is forced here (mirroring the SSE3 kernel
// dispatch tests).
//
//nolint:paralleltest // modifies global CPU feature state
func TestRecombineForwardSSE3MatchesGeneric(t *testing.T) {
	requireSSE3(t) // forcing HasSSE3 past dispatch would SIGILL on a non-SSE3 host

	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	cpu.SetForcedFeatures(cpu.Features{HasSSE: true, HasSSE2: true, HasSSE3: true})

	for _, half := range []int{2, 3, 4, 5, 8, 9, 13, 16, 17, 32, 33, 100, 129, 512, 1000} {
		t.Run(fmt.Sprintf("complex64/half=%d", half), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(half)))

			src := make([]complex64, half)
			for i := range src {
				src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
			}

			weight := recombineWeights64(half)

			got := make([]complex64, half+1)
			RecombineForwardComplex64(got, src, weight)

			want := make([]complex64, half+1)
			recombineForwardComplex64Generic(want, src, weight, 1)

			const eps = 1e-5

			for k := 1; k < half; k++ {
				dr := math.Abs(float64(real(got[k]) - real(want[k])))
				di := math.Abs(float64(imag(got[k]) - imag(want[k])))

				if dr > eps || di > eps {
					t.Errorf("bin %d: got %v, want %v", k, got[k], want[k])
				}
			}
		})

		t.Run(fmt.Sprintf("complex128/half=%d", half), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(half) + 2048))

			src := make([]complex128, half)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			weight := recombineWeights128(half)

			got := make([]complex128, half+1)
			RecombineForwardComplex128(got, src, weight)

			want := make([]complex128, half+1)
			recombineForwardComplex128Generic(want, src, weight, 1)

			const eps = 1e-14

			for k := 1; k < half; k++ {
				dr := math.Abs(real(got[k]) - real(want[k]))
				di := math.Abs(imag(got[k]) - imag(want[k]))

				if dr > eps || di > eps {
					t.Errorf("bin %d: got %v, want %v", k, got[k], want[k])
				}
			}
		})
	}
}
