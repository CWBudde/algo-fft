package fft

import (
	stdmath "math"
	"strconv"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// padShapeSizes are Bluestein pad candidates stratified by *shape* within four
// dyadic ranges. Each range ends in its power-of-two endpoint (the incumbent
// pad), preceded by mixed-radix-smooth candidates that span the two axes the
// mixed-radix engine is sensitive to: the size of the power-of-two part (which
// decides whether the schedule ends in a tuned codelet leaf) and whether the
// odd part needs a radix-7/11 full-matrix butterfly.
//
// Comments give the factorization as pow2 x odd.
//
//nolint:gochecknoglobals // benchmark input table
var padShapeSizes = []int{
	// (512, 1024]
	540,  // 4 x 135
	616,  // 8 x 77      (7*11)
	640,  // 128 x 5
	675,  // 1 x 675
	704,  // 64 x 11
	768,  // 256 x 3
	784,  // 16 x 49     (7^2)
	896,  // 128 x 7
	900,  // 4 x 225
	960,  // 64 x 15
	1024, // pow2
	// (2048, 4096]
	2160, // 16 x 135
	2250, // 2 x 1125
	2464, // 32 x 77     (7*11)
	2560, // 512 x 5
	2816, // 256 x 11
	3000, // 8 x 375
	3072, // 1024 x 3
	3136, // 64 x 49     (7^2)
	3584, // 512 x 7
	3888, // 16 x 243
	4096, // pow2
	// (8192, 16384]
	8640,  // 64 x 135
	9856,  // 128 x 77    (7*11)
	10240, // 2048 x 5
	11264, // 1024 x 11
	12000, // 32 x 375
	12288, // 4096 x 3
	12544, // 256 x 49    (7^2)
	14336, // 2048 x 7
	15360, // 1024 x 15
	16384, // pow2
	// (32768, 65536]
	34560, // 256 x 135
	39424, // 512 x 77    (7*11)
	40960, // 8192 x 5
	45056, // 4096 x 11
	49152, // 16384 x 3
	50176, // 1024 x 49   (7^2)
	51200, // 2048 x 25
	57344, // 8192 x 7
	61440, // 4096 x 15
	65536, // pow2
}

// BenchmarkBluesteinPadShapes measures one full Bluestein cyclic convolution
// (forward sub-FFT + filter multiply + inverse sub-FFT) at each padded size in
// padShapeSizes, for both precisions.
//
// This is the calibration benchmark for the shape-aware pad cost model in the
// root package (bluesteinPadPenalty / cheapestPaddedLength, PLAN.md P4.1). The
// quantity of interest is ns/op divided by m*log2(m): normalized against the
// power-of-two endpoint of the same dyadic range it is exactly the per-point
// penalty the cost model must encode. Run on the default build *and*
// -tags purego before changing that model.
//
// The sub-benchmark name carries m and the power-of-two part so the fit can be
// done straight from the benchstat output.
func BenchmarkBluesteinPadShapes(b *testing.B) {
	b.Run("complex64", func(b *testing.B) { benchPadSizes[complex64](b, padShapeSizes) })
	b.Run("complex128", func(b *testing.B) { benchPadSizes[complex128](b, padShapeSizes) })
}

func benchPadSizes[T Complex](b *testing.B, sizes []int) {
	b.Helper()

	for _, m := range sizes {
		b.Run(padShapeName(m), func(b *testing.B) {
			x := make([]T, m)
			for i := range x {
				x[i] = T(complex(float64(i%7)-3, float64(i%5)-2))
			}

			filter := make([]T, m)
			for i := range filter {
				filter[i] = T(complex(1, 0))
			}

			twiddle := mathpkg.ComputeTwiddleFactors[T](m)
			scratch := make([]T, m)
			work := make([]T, m)
			copy(work, x)

			var bitrev []int
			if mathpkg.IsPowerOf2(m) {
				bitrev = mathpkg.ComputeBitReversalIndices(m)
			}

			// Warm the mixed-radix scratch pools so steady state is measured.
			BluesteinConvolution(work, work, filter, twiddle, scratch, bitrev, nil)

			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				BluesteinConvolution(work, work, filter, twiddle, scratch, bitrev, nil)
			}

			// Report the model's cost unit so the penalty can be read off
			// directly: ns/op per m*log2(m) point-passes.
			b.ReportMetric(
				float64(b.Elapsed().Nanoseconds())/float64(b.N)/(float64(m)*stdmath.Log2(float64(m))),
				"ns/mlogm",
			)
		})
	}
}

// padFamilySizes covers the three candidate families a pad chooser that ranks
// by power-of-two depth can ever select, across every dyadic window from 2^7
// to 2^16 — including the windows BenchmarkBluesteinPadShapes leaves out.
//
// For a window ending at P = 2^k the deepest-power-of-two candidate below P is
// 3*2^(k-2) = 0.75P; when the required minimum excludes it the next-deepest is
// 7*2^(k-3) = 0.875P, then 15*2^(k-4) = 0.9375P. Every other smooth candidate
// in the window is both shallower and larger, so it can never be picked. Each
// window's power-of-two endpoint is included as the baseline to beat.
//
//nolint:gochecknoglobals // benchmark input table
var padFamilySizes = []int{
	96, 112, 120, 128, // P = 2^7
	192, 224, 240, 256, // P = 2^8
	384, 448, 480, 512, // P = 2^9
	768, 896, 960, 1024, // P = 2^10
	1536, 1792, 1920, 2048, // P = 2^11
	3072, 3584, 3840, 4096, // P = 2^12
	6144, 7168, 7680, 8192, // P = 2^13
	12288, 14336, 15360, 16384, // P = 2^14
	24576, 28672, 30720, 32768, // P = 2^15
	49152, 57344, 61440, 65536, // P = 2^16
}

// BenchmarkBluesteinPadFamilies is the second calibration pass: it measures the
// 0.75P / 0.875P / 0.9375P candidate families against their power-of-two
// endpoint in every window, so the accept threshold per family is read off
// measurements rather than extrapolated from the windows
// BenchmarkBluesteinPadShapes happens to cover.
func BenchmarkBluesteinPadFamilies(b *testing.B) {
	b.Run("complex64", func(b *testing.B) { benchPadSizes[complex64](b, padFamilySizes) })
	b.Run("complex128", func(b *testing.B) { benchPadSizes[complex128](b, padFamilySizes) })
}

// padShapeName encodes m and its power-of-two part, plus a marker for odd parts
// needing a radix-7/11 butterfly, so shapes group readably in benchstat output.
func padShapeName(m int) string {
	if mathpkg.IsPowerOf2(m) {
		return "m" + strconv.Itoa(m) + "_pow2"
	}

	name := "m" + strconv.Itoa(m) + "_p2x" + strconv.Itoa(m&-m)
	if !mathpkg.IsHighlyComposite(m) {
		name += "_odd711"
	}

	return name
}
