package algofft

import (
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// Forward FFT benchmarks for various sizes.
func BenchmarkPlanForward_8(b *testing.B)     { benchmarkPlanForward(b, 8) }
func BenchmarkPlanForward_16(b *testing.B)    { benchmarkPlanForward(b, 16) }
func BenchmarkPlanForward_32(b *testing.B)    { benchmarkPlanForward(b, 32) }
func BenchmarkPlanForward_64(b *testing.B)    { benchmarkPlanForward(b, 64) }
func BenchmarkPlanForward_128(b *testing.B)   { benchmarkPlanForward(b, 128) }
func BenchmarkPlanForward_256(b *testing.B)   { benchmarkPlanForward(b, 256) }
func BenchmarkPlanForward_1024(b *testing.B)  { benchmarkPlanForward(b, 1024) }
func BenchmarkPlanForward_4096(b *testing.B)  { benchmarkPlanForward(b, 4096) }
func BenchmarkPlanForward_8192(b *testing.B)  { benchmarkPlanForward(b, 8192) }
func BenchmarkPlanForward_16384(b *testing.B) { benchmarkPlanForward(b, 16384) }
func BenchmarkPlanForward_65536(b *testing.B) { benchmarkPlanForward(b, 65536) }
func BenchmarkPlanForward_8192_DIT(b *testing.B) {
	benchmarkPlanForwardWithOptions(b, 8192, PlanOptions{Strategy: KernelDIT})
}
func BenchmarkPlanForward_8192_Stockham(b *testing.B) {
	benchmarkPlanForwardWithOptions(b, 8192, PlanOptions{Strategy: KernelStockham})
}

// Inverse FFT benchmarks for various sizes.
func BenchmarkPlanInverse_8(b *testing.B)     { benchmarkPlanInverse(b, 8) }
func BenchmarkPlanInverse_16(b *testing.B)    { benchmarkPlanInverse(b, 16) }
func BenchmarkPlanInverse_32(b *testing.B)    { benchmarkPlanInverse(b, 32) }
func BenchmarkPlanInverse_64(b *testing.B)    { benchmarkPlanInverse(b, 64) }
func BenchmarkPlanInverse_128(b *testing.B)   { benchmarkPlanInverse(b, 128) }
func BenchmarkPlanInverse_256(b *testing.B)   { benchmarkPlanInverse(b, 256) }
func BenchmarkPlanInverse_1024(b *testing.B)  { benchmarkPlanInverse(b, 1024) }
func BenchmarkPlanInverse_4096(b *testing.B)  { benchmarkPlanInverse(b, 4096) }
func BenchmarkPlanInverse_8192(b *testing.B)  { benchmarkPlanInverse(b, 8192) }
func BenchmarkPlanInverse_16384(b *testing.B) { benchmarkPlanInverse(b, 16384) }
func BenchmarkPlanInverse_65536(b *testing.B) { benchmarkPlanInverse(b, 65536) }
func BenchmarkPlanInverse_8192_DIT(b *testing.B) {
	benchmarkPlanInverseWithOptions(b, 8192, PlanOptions{Strategy: KernelDIT})
}
func BenchmarkPlanInverse_8192_Stockham(b *testing.B) {
	benchmarkPlanInverseWithOptions(b, 8192, PlanOptions{Strategy: KernelStockham})
}

// Plan creation benchmarks (additional sizes - 64, 1024, 65536 are in plan_test.go).
func BenchmarkNewPlan_16(b *testing.B)    { benchmarkNewPlan(b, 16) }
func BenchmarkNewPlan_256(b *testing.B)   { benchmarkNewPlan(b, 256) }
func BenchmarkNewPlan_4096(b *testing.B)  { benchmarkNewPlan(b, 4096) }
func BenchmarkNewPlan_16384(b *testing.B) { benchmarkNewPlan(b, 16384) }

// Reference DFT comparison benchmarks (for small sizes only due to O(n²) complexity).
func BenchmarkReferenceDFT_16(b *testing.B)  { benchmarkReferenceDFT(b, 16) }
func BenchmarkReferenceDFT_64(b *testing.B)  { benchmarkReferenceDFT(b, 64) }
func BenchmarkReferenceDFT_256(b *testing.B) { benchmarkReferenceDFT(b, 256) }

// Memory profiling benchmark using runtime.MemStats.
func BenchmarkPlanForward_MemStats_1024(b *testing.B) { benchmarkPlanForwardMemStats(b, 1024) }

// Benchmark transform behavior under different plan/buffer reuse patterns.
func BenchmarkPlanReusePatterns_1024(b *testing.B) {
	b.Run("ReusePlanReuseBuffers", func(b *testing.B) {
		benchmarkPlanForward(b, 1024)
	})
	b.Run("ReusePlanAllocBuffers", func(b *testing.B) {
		benchmarkPlanForwardAllocBuffers(b, 1024)
	})
	b.Run("NewPlanEachIter", func(b *testing.B) {
		benchmarkPlanForwardNewPlanEachIter(b, 1024)
	})
}

func benchmarkPlanForward(b *testing.B, fftSize int) {
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
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		fwdErr := plan.Forward(dst, src)
		if fwdErr != nil {
			b.Fatalf("Forward() returned error: %v", fwdErr)
		}
	}
}

func benchmarkPlanForwardWithOptions(b *testing.B, fftSize int, opts PlanOptions) {
	b.Helper()

	plan, err := NewPlanWithOptions[complex64](fftSize, opts)
	if err != nil {
		b.Fatalf("NewPlanWithOptions(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, fftSize)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		fwdErr := plan.Forward(dst, src)
		if fwdErr != nil {
			b.Fatalf("Forward() returned error: %v", fwdErr)
		}
	}
}

func benchmarkPlanForwardMemStats(b *testing.B, fftSize int) {
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

	runtime.GC()

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		err := plan.Forward(dst, src)
		if err != nil {
			b.Fatalf("Forward() returned error: %v", err)
		}
	}

	b.StopTimer()
	runtime.ReadMemStats(&after)

	allocBytes := int64(after.TotalAlloc - before.TotalAlloc)
	allocs := int64(after.Mallocs - before.Mallocs)

	b.ReportMetric(float64(allocBytes)/float64(b.N), "bytes/op")
	b.ReportMetric(float64(allocs)/float64(b.N), "allocs/op")
}

func benchmarkPlanForwardAllocBuffers(b *testing.B, fftSize int) {
	b.Helper()

	plan, err := NewPlanT[complex64](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	template := make([]complex64, fftSize)
	for i := range template {
		template[i] = complex(float32(i+1), float32(-i))
	}

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		src := make([]complex64, fftSize)
		copy(src, template)

		dst := make([]complex64, fftSize)

		err := plan.Forward(dst, src)
		if err != nil {
			b.Fatalf("Forward() returned error: %v", err)
		}
	}
}

func benchmarkPlanForwardNewPlanEachIter(b *testing.B, fftSize int) {
	b.Helper()

	template := make([]complex64, fftSize)
	for i := range template {
		template[i] = complex(float32(i+1), float32(-i))
	}

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		plan, err := NewPlanT[complex64](fftSize)
		if err != nil {
			b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
		}

		src := make([]complex64, fftSize)
		copy(src, template)

		dst := make([]complex64, fftSize)

		err = plan.Forward(dst, src)
		if err != nil {
			b.Fatalf("Forward() returned error: %v", err)
		}
	}
}

func benchmarkPlanInverse(b *testing.B, fftSize int) {
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
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		invErr := plan.Inverse(dst, freq)
		if invErr != nil {
			b.Fatalf("Inverse() returned error: %v", invErr)
		}
	}
}

func benchmarkPlanInverseWithOptions(b *testing.B, fftSize int, opts PlanOptions) {
	b.Helper()

	plan, err := NewPlanWithOptions[complex64](fftSize, opts)
	if err != nil {
		b.Fatalf("NewPlanWithOptions(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, fftSize)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		invErr := plan.Inverse(dst, src)
		if invErr != nil {
			b.Fatalf("Inverse() returned error: %v", invErr)
		}
	}
}

func benchmarkNewPlan(b *testing.B, fftSize int) {
	b.Helper()
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		plan, err := NewPlanT[complex64](fftSize)
		if err != nil {
			b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
		}

		// Prevent compiler from optimizing away the allocation
		_ = plan
	}
}

func benchmarkReferenceDFT(b *testing.B, fftSize int) {
	b.Helper()

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		result := reference.NaiveDFT(src)
		// Prevent compiler from optimizing away the result
		_ = result
	}
}
