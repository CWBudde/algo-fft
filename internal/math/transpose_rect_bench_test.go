package math

import (
	"fmt"
	"testing"
)

func benchTransposeRect[T any](b *testing.B, rows, cols, elemSize int) {
	b.Helper()

	src := make([]T, rows*cols)
	dst := make([]T, rows*cols)

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(rows * cols * elemSize))

	for b.Loop() {
		TransposeRect(dst, src, rows, cols)
	}
}

func BenchmarkTransposeRect(b *testing.B) {
	shapes := []struct{ rows, cols int }{
		{256, 256}, {256, 1024}, {1024, 256}, {512, 2048}, {1024, 1024},
	}
	for _, s := range shapes {
		b.Run(fmt.Sprintf("Complex64/%dx%d", s.rows, s.cols), func(b *testing.B) {
			benchTransposeRect[complex64](b, s.rows, s.cols, 8)
		})
		b.Run(fmt.Sprintf("Complex128/%dx%d", s.rows, s.cols), func(b *testing.B) {
			benchTransposeRect[complex128](b, s.rows, s.cols, 16)
		})
	}
}
