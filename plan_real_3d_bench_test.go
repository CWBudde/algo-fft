package algoforge

import (
	"testing"
)

func BenchmarkPlanReal3D_Forward(b *testing.B) {
	sizes := []struct {
		depth, height, width int
	}{
		{8, 8, 8},
		{16, 16, 16},
		{32, 32, 32},
		{64, 64, 64},
		{128, 128, 128},
	}

	for _, size := range sizes {
		b.Run(sprintf3d(size.depth, size.height, size.width), func(b *testing.B) {
			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				b.Fatalf("NewPlanReal3D failed: %v", err)
			}

			input := make([]float32, size.depth*size.height*size.width)
			spectrum := make([]complex64, plan.SpectrumLen())

			// Initialize with random data
			for i := range input {
				input[i] = float32(i % 100)
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(len(input) * 4)) // 4 bytes per float32

			for i := 0; i < b.N; i++ {
				_ = plan.Forward(spectrum, input)
			}
		})
	}
}

func BenchmarkPlanReal3D_ForwardFull(b *testing.B) {
	sizes := []struct {
		depth, height, width int
	}{
		{8, 8, 8},
		{16, 16, 16},
		{32, 32, 32},
		{64, 64, 64},
	}

	for _, size := range sizes {
		b.Run(sprintf3d(size.depth, size.height, size.width), func(b *testing.B) {
			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				b.Fatalf("NewPlanReal3D failed: %v", err)
			}

			input := make([]float32, size.depth*size.height*size.width)
			spectrum := make([]complex64, size.depth*size.height*size.width)

			for i := range input {
				input[i] = float32(i % 100)
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(len(input) * 4))

			for i := 0; i < b.N; i++ {
				_ = plan.ForwardFull(spectrum, input)
			}
		})
	}
}

func BenchmarkPlanReal3D_Inverse(b *testing.B) {
	sizes := []struct {
		depth, height, width int
	}{
		{8, 8, 8},
		{16, 16, 16},
		{32, 32, 32},
		{64, 64, 64},
		{128, 128, 128},
	}

	for _, size := range sizes {
		b.Run(sprintf3d(size.depth, size.height, size.width), func(b *testing.B) {
			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				b.Fatalf("NewPlanReal3D failed: %v", err)
			}

			spectrum := make([]complex64, plan.SpectrumLen())
			output := make([]float32, size.depth*size.height*size.width)

			for i := range spectrum {
				spectrum[i] = complex(float32(i%100), float32((i+1)%100))
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(len(output) * 4))

			for i := 0; i < b.N; i++ {
				_ = plan.Inverse(output, spectrum)
			}
		})
	}
}

func BenchmarkPlanReal3D_RoundTrip(b *testing.B) {
	sizes := []struct {
		depth, height, width int
	}{
		{8, 8, 8},
		{16, 16, 16},
		{32, 32, 32},
		{64, 64, 64},
	}

	for _, size := range sizes {
		b.Run(sprintf3d(size.depth, size.height, size.width), func(b *testing.B) {
			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				b.Fatalf("NewPlanReal3D failed: %v", err)
			}

			input := make([]float32, size.depth*size.height*size.width)
			spectrum := make([]complex64, plan.SpectrumLen())
			output := make([]float32, size.depth*size.height*size.width)

			for i := range input {
				input[i] = float32(i % 100)
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(len(input) * 4))

			for i := 0; i < b.N; i++ {
				_ = plan.Forward(spectrum, input)
				_ = plan.Inverse(output, spectrum)
			}
		})
	}
}

func BenchmarkPlanReal3D_NonCubic(b *testing.B) {
	sizes := []struct {
		depth, height, width int
	}{
		{8, 8, 16},
		{8, 16, 16},
		{16, 16, 32},
		{16, 32, 32},
	}

	for _, size := range sizes {
		b.Run(sprintf3d(size.depth, size.height, size.width), func(b *testing.B) {
			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				b.Fatalf("NewPlanReal3D failed: %v", err)
			}

			input := make([]float32, size.depth*size.height*size.width)
			spectrum := make([]complex64, plan.SpectrumLen())

			for i := range input {
				input[i] = float32(i % 100)
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(len(input) * 4))

			for i := 0; i < b.N; i++ {
				_ = plan.Forward(spectrum, input)
			}
		})
	}
}
