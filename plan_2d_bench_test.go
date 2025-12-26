package algoforge

import (
	"testing"
)

// Benchmark helpers

func benchmarkPlan2DForward(b *testing.B, rows, cols int) {
	b.Helper()
	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		b.Fatalf("NewPlan2D failed: %v", err)
	}

	src := generateRandom2DSignal(rows, cols, 12345)
	dst := make([]complex64, rows*cols)

	b.ReportAllocs()
	b.SetBytes(int64(rows * cols * 8)) // complex64 = 8 bytes
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = plan.Forward(dst, src)
	}
}

func benchmarkPlan2DInverse(b *testing.B, rows, cols int) {
	b.Helper()
	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		b.Fatalf("NewPlan2D failed: %v", err)
	}

	src := generateRandom2DSignal(rows, cols, 12345)
	dst := make([]complex64, rows*cols)

	b.ReportAllocs()
	b.SetBytes(int64(rows * cols * 8))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = plan.Inverse(dst, src)
	}
}

func benchmarkPlan2DInPlace(b *testing.B, rows, cols int) {
	b.Helper()
	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		b.Fatalf("NewPlan2D failed: %v", err)
	}

	data := generateRandom2DSignal(rows, cols, 12345)

	b.ReportAllocs()
	b.SetBytes(int64(rows * cols * 8))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = plan.ForwardInPlace(data)
	}
}

// Square matrices

func BenchmarkPlan2D_Forward_8x8(b *testing.B) {
	benchmarkPlan2DForward(b, 8, 8)
}

func BenchmarkPlan2D_Forward_16x16(b *testing.B) {
	benchmarkPlan2DForward(b, 16, 16)
}

func BenchmarkPlan2D_Forward_32x32(b *testing.B) {
	benchmarkPlan2DForward(b, 32, 32)
}

func BenchmarkPlan2D_Forward_64x64(b *testing.B) {
	benchmarkPlan2DForward(b, 64, 64)
}

func BenchmarkPlan2D_Forward_128x128(b *testing.B) {
	benchmarkPlan2DForward(b, 128, 128)
}

func BenchmarkPlan2D_Forward_256x256(b *testing.B) {
	benchmarkPlan2DForward(b, 256, 256)
}

func BenchmarkPlan2D_Forward_512x512(b *testing.B) {
	benchmarkPlan2DForward(b, 512, 512)
}

// Inverse transforms

func BenchmarkPlan2D_Inverse_8x8(b *testing.B) {
	benchmarkPlan2DInverse(b, 8, 8)
}

func BenchmarkPlan2D_Inverse_16x16(b *testing.B) {
	benchmarkPlan2DInverse(b, 16, 16)
}

func BenchmarkPlan2D_Inverse_32x32(b *testing.B) {
	benchmarkPlan2DInverse(b, 32, 32)
}

func BenchmarkPlan2D_Inverse_64x64(b *testing.B) {
	benchmarkPlan2DInverse(b, 64, 64)
}

func BenchmarkPlan2D_Inverse_128x128(b *testing.B) {
	benchmarkPlan2DInverse(b, 128, 128)
}

func BenchmarkPlan2D_Inverse_256x256(b *testing.B) {
	benchmarkPlan2DInverse(b, 256, 256)
}

// Non-square matrices

func BenchmarkPlan2D_Forward_16x32(b *testing.B) {
	benchmarkPlan2DForward(b, 16, 32)
}

func BenchmarkPlan2D_Forward_32x64(b *testing.B) {
	benchmarkPlan2DForward(b, 32, 64)
}

func BenchmarkPlan2D_Forward_64x128(b *testing.B) {
	benchmarkPlan2DForward(b, 64, 128)
}

func BenchmarkPlan2D_Forward_128x256(b *testing.B) {
	benchmarkPlan2DForward(b, 128, 256)
}

// In-place vs out-of-place

func BenchmarkPlan2D_InPlaceVsOutOfPlace_64x64(b *testing.B) {
	b.Run("OutOfPlace", func(b *testing.B) {
		benchmarkPlan2DForward(b, 64, 64)
	})

	b.Run("InPlace", func(b *testing.B) {
		benchmarkPlan2DInPlace(b, 64, 64)
	})
}

// Plan reuse patterns

func BenchmarkPlan2D_ReusePatterns_64x64(b *testing.B) {
	b.Run("ReusePlanReuseBuffers", func(b *testing.B) {
		benchmarkPlan2DForward(b, 64, 64)
	})

	b.Run("ReusePlanAllocBuffers", func(b *testing.B) {
		plan, err := NewPlan2D[complex64](64, 64)
		if err != nil {
			b.Fatalf("NewPlan2D failed: %v", err)
		}

		src := generateRandom2DSignal(64, 64, 12345)

		b.ReportAllocs()
		b.SetBytes(int64(64 * 64 * 8))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			dst := make([]complex64, 64*64)
			_ = plan.Forward(dst, src)
		}
	})

	b.Run("NewPlanEachIter", func(b *testing.B) {
		src := generateRandom2DSignal(64, 64, 12345)
		dst := make([]complex64, 64*64)

		b.ReportAllocs()
		b.SetBytes(int64(64 * 64 * 8))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			plan, _ := NewPlan2D[complex64](64, 64)
			_ = plan.Forward(dst, src)
		}
	})
}

// Complex128 precision

func BenchmarkPlan2D_Forward_64x64_Complex128(b *testing.B) {
	plan, err := NewPlan2D[complex128](64, 64)
	if err != nil {
		b.Fatalf("NewPlan2D failed: %v", err)
	}

	src := generateRandom2DSignal128(64, 64, 12345)
	dst := make([]complex128, 64*64)

	b.ReportAllocs()
	b.SetBytes(int64(64 * 64 * 16)) // complex128 = 16 bytes
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = plan.Forward(dst, src)
	}
}
