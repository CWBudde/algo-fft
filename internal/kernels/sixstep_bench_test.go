package kernels

import (
	"fmt"
	"testing"
)

// BenchmarkSixStepComplex64 benchmarks the generic six-step kernel for
// square power-of-two sizes (m×m with m = sqrt(n)).
func BenchmarkSixStepComplex64(b *testing.B) {
	for _, n := range []int{4096, 65536, 262144, 1048576} {
		b.Run(fmt.Sprintf("Size%d/Forward", n), func(b *testing.B) {
			runBenchComplex64(b, n, ForwardSixStepComplex64)
		})
		b.Run(fmt.Sprintf("Size%d/Inverse", n), func(b *testing.B) {
			runBenchComplex64(b, n, InverseSixStepComplex64)
		})
	}
}

// BenchmarkSixStepComplex128 benchmarks the generic six-step kernel for
// square power-of-two sizes (m×m with m = sqrt(n)).
func BenchmarkSixStepComplex128(b *testing.B) {
	for _, n := range []int{4096, 65536, 262144, 1048576} {
		b.Run(fmt.Sprintf("Size%d/Forward", n), func(b *testing.B) {
			runBenchComplex128(b, n, ForwardSixStepComplex128)
		})
		b.Run(fmt.Sprintf("Size%d/Inverse", n), func(b *testing.B) {
			runBenchComplex128(b, n, InverseSixStepComplex128)
		})
	}
}
