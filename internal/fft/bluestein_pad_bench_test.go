package fft

import (
	"strconv"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// BenchmarkBluesteinPadCandidates measures one full Bluestein cyclic
// convolution (forward sub-FFT + filter multiply + inverse sub-FFT) at
// candidate padded sizes m: powers of two run the size-dispatched DIT
// kernels, other 5-smooth sizes run the mixed-radix engine.
//
// It predates the shape-aware pad model and is kept as a spot check for
// individual sizes; BenchmarkBluesteinPadShapes and
// BenchmarkBluesteinPadFamilies are the calibration benchmarks for padShapes
// in the root package (see PLAN.md P4.1).
func BenchmarkBluesteinPadCandidates(b *testing.B) {
	sizes := []int{24, 32, 45, 64, 540, 1000, 1024, 1440, 2000, 2025, 2048, 4050, 4096, 6075, 8192}

	for _, m := range sizes {
		kind := "smooth"
		if mathpkg.IsPowerOf2(m) {
			kind = "pow2"
		}

		b.Run(kind+"_"+strconv.Itoa(m), func(b *testing.B) {
			x := make([]complex64, m)
			for i := range x {
				x[i] = complex(float32(i%7)-3, float32(i%5)-2)
			}

			filter := make([]complex64, m)
			for i := range filter {
				filter[i] = complex(1, 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](m)
			scratch := make([]complex64, m)
			work := make([]complex64, m)
			copy(work, x)

			var bitrev []int
			if mathpkg.IsPowerOf2(m) {
				bitrev = mathpkg.ComputeBitReversalIndices(m)
			}

			// Warm the mixed-radix scratch pools so steady-state is measured.
			BluesteinConvolution(work, work, filter, twiddle, scratch, bitrev)

			b.ReportAllocs()
			b.SetBytes(int64(m * 8))
			b.ResetTimer()

			for range b.N {
				BluesteinConvolution(work, work, filter, twiddle, scratch, bitrev)
			}
		})
	}
}
