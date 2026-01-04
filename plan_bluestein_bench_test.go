package algofft

import (
	"testing"
)

// Bluestein FFT benchmarks for non-power-of-2 sizes.
// These use Bluestein's algorithm with internal power-of-2 FFTs.
//
// Size 384 = 2^7 × 3, internal FFT size = 1024 (NextPowerOfTwo(2*384-1) = 1024)
// Size 768 = 2^8 × 3, internal FFT size = 2048 (NextPowerOfTwo(2*768-1) = 2048)
// Size 1000, internal FFT size = 2048 (NextPowerOfTwo(2*1000-1) = 2048)
// Size 1536 = 2^9 × 3, internal FFT size = 4096 (NextPowerOfTwo(2*1536-1) = 4096)
// Size 3000, internal FFT size = 8192 (NextPowerOfTwo(2*3000-1) = 8192)

// Forward Bluestein benchmarks.
func BenchmarkBluestein_Forward_384(b *testing.B)  { benchmarkBluesteinForward(b, 384) }
func BenchmarkBluestein_Forward_768(b *testing.B)  { benchmarkBluesteinForward(b, 768) }
func BenchmarkBluestein_Forward_1000(b *testing.B) { benchmarkBluesteinForward(b, 1000) }
func BenchmarkBluestein_Forward_1536(b *testing.B) { benchmarkBluesteinForward(b, 1536) }
func BenchmarkBluestein_Forward_3000(b *testing.B) { benchmarkBluesteinForward(b, 3000) }

// Inverse Bluestein benchmarks.
func BenchmarkBluestein_Inverse_384(b *testing.B)  { benchmarkBluesteinInverse(b, 384) }
func BenchmarkBluestein_Inverse_768(b *testing.B)  { benchmarkBluesteinInverse(b, 768) }
func BenchmarkBluestein_Inverse_1000(b *testing.B) { benchmarkBluesteinInverse(b, 1000) }
func BenchmarkBluestein_Inverse_1536(b *testing.B) { benchmarkBluesteinInverse(b, 1536) }
func BenchmarkBluestein_Inverse_3000(b *testing.B) { benchmarkBluesteinInverse(b, 3000) }

// Plan creation benchmarks for Bluestein sizes.
func BenchmarkBluestein_NewPlan_384(b *testing.B)  { benchmarkBluesteinNewPlan(b, 384) }
func BenchmarkBluestein_NewPlan_768(b *testing.B)  { benchmarkBluesteinNewPlan(b, 768) }
func BenchmarkBluestein_NewPlan_1000(b *testing.B) { benchmarkBluesteinNewPlan(b, 1000) }
func BenchmarkBluestein_NewPlan_1536(b *testing.B) { benchmarkBluesteinNewPlan(b, 1536) }
func BenchmarkBluestein_NewPlan_3000(b *testing.B) { benchmarkBluesteinNewPlan(b, 3000) }

// Complex128 benchmarks (higher precision).
func BenchmarkBluestein_Forward_384_Complex128(b *testing.B)  { benchmarkBluesteinForward128(b, 384) }
func BenchmarkBluestein_Forward_1000_Complex128(b *testing.B) { benchmarkBluesteinForward128(b, 1000) }
func BenchmarkBluestein_Forward_3000_Complex128(b *testing.B) { benchmarkBluesteinForward128(b, 3000) }

// Comparison benchmarks: Bluestein vs nearest power-of-2
// This helps understand the overhead of Bluestein vs padding to next power of 2.
func BenchmarkComparison_384_vs_512(b *testing.B) {
	b.Run("Bluestein_384", func(b *testing.B) { benchmarkBluesteinForward(b, 384) })
	b.Run("PowerOf2_512", func(b *testing.B) { benchmarkPlanForward(b, 512) })
}

func BenchmarkComparison_1000_vs_1024(b *testing.B) {
	b.Run("Bluestein_1000", func(b *testing.B) { benchmarkBluesteinForward(b, 1000) })
	b.Run("PowerOf2_1024", func(b *testing.B) { benchmarkPlanForward(b, 1024) })
}

func BenchmarkComparison_3000_vs_4096(b *testing.B) {
	b.Run("Bluestein_3000", func(b *testing.B) { benchmarkBluesteinForward(b, 3000) })
	b.Run("PowerOf2_4096", func(b *testing.B) { benchmarkPlanForward(b, 4096) })
}

func benchmarkBluesteinForward(b *testing.B, fftSize int) {
	b.Helper()

	plan, err := NewPlanT[complex64](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, fftSize)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64
	b.ResetTimer()

	for b.Loop() {
		fwdErr := plan.Forward(dst, src)
		if fwdErr != nil {
			b.Fatalf("Forward() returned error: %v", fwdErr)
		}
	}
}

func benchmarkBluesteinInverse(b *testing.B, fftSize int) {
	b.Helper()

	plan, err := NewPlanT[complex64](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	freq := make([]complex64, fftSize)

	fwdErr := plan.Forward(freq, src)
	if fwdErr != nil {
		b.Fatalf("Forward() returned error: %v", fwdErr)
	}

	dst := make([]complex64, fftSize)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64
	b.ResetTimer()

	for b.Loop() {
		invErr := plan.Inverse(dst, freq)
		if invErr != nil {
			b.Fatalf("Inverse() returned error: %v", invErr)
		}
	}
}

func benchmarkBluesteinNewPlan(b *testing.B, fftSize int) {
	b.Helper()
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		plan, err := NewPlanT[complex64](fftSize)
		if err != nil {
			b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
		}

		_ = plan
	}
}

func benchmarkBluesteinForward128(b *testing.B, fftSize int) {
	b.Helper()

	plan, err := NewPlanT[complex128](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex128, fftSize)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, fftSize)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 16)) // 16 bytes per complex128
	b.ResetTimer()

	for b.Loop() {
		fwdErr := plan.Forward(dst, src)
		if fwdErr != nil {
			b.Fatalf("Forward() returned error: %v", fwdErr)
		}
	}
}
