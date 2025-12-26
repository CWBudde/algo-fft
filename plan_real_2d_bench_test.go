package algoforge

import (
	"testing"
)

func BenchmarkPlanReal2D_Forward(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{8, 8},
		{16, 16},
		{32, 32},
		{64, 64},
		{128, 128},
		{256, 256},
		{512, 512},
	}

	for _, size := range sizes {
		b.Run(sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				b.Fatalf("NewPlanReal2D failed: %v", err)
			}

			input := make([]float32, size.rows*size.cols)
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

func BenchmarkPlanReal2D_ForwardFull(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{8, 8},
		{16, 16},
		{32, 32},
		{64, 64},
		{128, 128},
		{256, 256},
	}

	for _, size := range sizes {
		b.Run(sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				b.Fatalf("NewPlanReal2D failed: %v", err)
			}

			input := make([]float32, size.rows*size.cols)
			spectrum := make([]complex64, size.rows*size.cols)

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

func BenchmarkPlanReal2D_Inverse(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{8, 8},
		{16, 16},
		{32, 32},
		{64, 64},
		{128, 128},
		{256, 256},
		{512, 512},
	}

	for _, size := range sizes {
		b.Run(sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				b.Fatalf("NewPlanReal2D failed: %v", err)
			}

			spectrum := make([]complex64, plan.SpectrumLen())
			output := make([]float32, size.rows*size.cols)

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

func BenchmarkPlanReal2D_RoundTrip(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{8, 8},
		{16, 16},
		{32, 32},
		{64, 64},
		{128, 128},
		{256, 256},
	}

	for _, size := range sizes {
		b.Run(sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				b.Fatalf("NewPlanReal2D failed: %v", err)
			}

			input := make([]float32, size.rows*size.cols)
			spectrum := make([]complex64, plan.SpectrumLen())
			output := make([]float32, size.rows*size.cols)

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

func BenchmarkPlanReal2D_NonSquare(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{16, 32},
		{32, 64},
		{64, 128},
		{128, 256},
	}

	for _, size := range sizes {
		b.Run(sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				b.Fatalf("NewPlanReal2D failed: %v", err)
			}

			input := make([]float32, size.rows*size.cols)
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
