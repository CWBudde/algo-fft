package math

import (
	"fmt"
	"testing"
)

func benchTransposeSquare[T any](b *testing.B, m, elemSize int) {
	b.Helper()

	data := make([]T, m*m)

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(m * m * elemSize))

	for b.Loop() {
		TransposeSquare(data, m)
	}
}

func BenchmarkTransposeSquare(b *testing.B) {
	for _, m := range []int{16, 64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("Complex64/%dx%d", m, m), func(b *testing.B) {
			benchTransposeSquare[complex64](b, m, 8)
		})
		b.Run(fmt.Sprintf("Complex128/%dx%d", m, m), func(b *testing.B) {
			benchTransposeSquare[complex128](b, m, 16)
		})
	}
}

// BenchmarkTransposeSquareBlockSize sweeps tile edges to justify the
// transposeBlock constant.
func BenchmarkTransposeSquareBlockSize(b *testing.B) {
	for _, block := range []int{4, 8, 16, 32} {
		for _, m := range []int{256, 1024} {
			b.Run(fmt.Sprintf("Block%d/Complex64/%dx%d", block, m, m), func(b *testing.B) {
				data := make([]complex64, m*m)

				b.ReportAllocs()
				b.SetBytes(int64(m * m * 8))

				for b.Loop() {
					transposeSquareBlocked(data, m, block)
				}
			})
			b.Run(fmt.Sprintf("Block%d/Complex128/%dx%d", block, m, m), func(b *testing.B) {
				data := make([]complex128, m*m)

				b.ReportAllocs()
				b.SetBytes(int64(m * m * 16))

				for b.Loop() {
					transposeSquareBlocked(data, m, block)
				}
			})
		}
	}
}
