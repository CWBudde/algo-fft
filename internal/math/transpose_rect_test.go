package math

import (
	"math/rand"
	"testing"
)

func TestTransposeRectSmall(t *testing.T) {
	// 2×3 matrix:
	//   1 2 3
	//   4 5 6
	src := []int{1, 2, 3, 4, 5, 6}
	dst := make([]int, 6)

	TransposeRect(dst, src, 2, 3)

	// Expect 3×2:
	//   1 4
	//   2 5
	//   3 6
	want := []int{1, 4, 2, 5, 3, 6}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst = %v, want %v", dst, want)
		}
	}
}

func TestTransposeRectMatchesNaive(t *testing.T) {
	shapes := []struct{ rows, cols int }{
		{1, 1},
		{1, 7},
		{7, 1},
		{2, 2},
		{3, 5},
		{8, 8},
		{9, 33},
		{32, 64},
		{64, 32},
		{128, 128},
		{128, 512},
		{512, 128},
		{100, 200},
		{31, 257},
	}

	rng := rand.New(rand.NewSource(42))

	for _, shape := range shapes {
		n := shape.rows * shape.cols

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(rng.Float32(), rng.Float32())
		}

		dst := make([]complex64, n)
		TransposeRect(dst, src, shape.rows, shape.cols)

		for r := range shape.rows {
			for c := range shape.cols {
				got := dst[c*shape.rows+r]

				want := src[r*shape.cols+c]
				if got != want {
					t.Fatalf("%dx%d: dst[%d][%d] = %v, want %v",
						shape.rows, shape.cols, c, r, got, want)
				}
			}
		}
	}
}

func TestTransposeRectRoundTrip(t *testing.T) {
	const rows, cols = 48, 96

	src := make([]complex128, rows*cols)
	for i := range src {
		src[i] = complex(float64(i), -float64(i))
	}

	mid := make([]complex128, rows*cols)
	dst := make([]complex128, rows*cols)

	TransposeRect(mid, src, rows, cols)
	TransposeRect(dst, mid, cols, rows)

	for i := range src {
		if dst[i] != src[i] {
			t.Fatalf("round trip mismatch at %d: got %v, want %v", i, dst[i], src[i])
		}
	}
}
