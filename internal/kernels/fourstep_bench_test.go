package kernels

import (
	"fmt"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// fourStepBenchSizes covers the above-L2 range the cache-blocked split
// targets (2^18 ... 2^23), including the non-square powers of two that
// six-step cannot handle.
//
//nolint:gochecknoglobals // static benchmark size table
var fourStepBenchSizes = []int{1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23}

// BenchmarkFourStepComplex64 benchmarks the four-step kernel with its
// cache-derived split.
func BenchmarkFourStepComplex64(b *testing.B) {
	for _, n := range fourStepBenchSizes {
		b.Run(fmt.Sprintf("Size%d/Forward", n), func(b *testing.B) {
			runBenchComplex64(b, n, ForwardFourStepComplex64)
		})
		b.Run(fmt.Sprintf("Size%d/Inverse", n), func(b *testing.B) {
			runBenchComplex64(b, n, InverseFourStepComplex64)
		})
	}
}

// BenchmarkFourStepComplex128 benchmarks the four-step kernel with its
// cache-derived split.
func BenchmarkFourStepComplex128(b *testing.B) {
	for _, n := range fourStepBenchSizes {
		b.Run(fmt.Sprintf("Size%d/Forward", n), func(b *testing.B) {
			runBenchComplex128(b, n, ForwardFourStepComplex128)
		})
		b.Run(fmt.Sprintf("Size%d/Inverse", n), func(b *testing.B) {
			runBenchComplex128(b, n, InverseFourStepComplex128)
		})
	}
}

// BenchmarkFourStepSplitSweep sweeps every n1×n2 split at each size so the
// cache-derived choice can be validated against the measured optimum.
func BenchmarkFourStepSplitSweep(b *testing.B) {
	caches := cpu.DetectCaches()

	for _, n := range []int{1 << 18, 1 << 20, 1 << 22} {
		chosenN1, chosenN2 := fourStepSplit(n, 8, caches)
		b.Logf("n=%d: cache-derived split %dx%d", n, chosenN1, chosenN2)

		for n1 := 32; n1 <= n/32; n1 *= 2 {
			n2 := n / n1
			b.Run(fmt.Sprintf("Size%d/%dx%d", n, n1, n2), func(b *testing.B) {
				benchFourStepSplit(b, n, n1)
			})
		}
	}
}

func benchFourStepSplit(b *testing.B, n, n1 int) {
	b.Helper()

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	for i := range src {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8))

	for b.Loop() {
		fourStepTransform(dst, src, twiddle, scratch, n1, false)
	}
}
