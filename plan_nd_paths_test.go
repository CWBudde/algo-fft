package algofft

import (
	"math/cmplx"
	"math/rand"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// These tests pin the PlanND engine against the naive reference DFT on shapes
// chosen to exercise every axis-transform path: contiguous innermost rows,
// square trailing 2D slabs (transpose path), and general strided axes
// (including odd, non-power-of-two sizes).

func randComplex64(n int, seed int64) []complex64 {
	rng := rand.New(rand.NewSource(seed))

	data := make([]complex64, n)
	for i := range data {
		data[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	return data
}

func TestPlanND_2D_MatchesReference(t *testing.T) {
	shapes := [][2]int{
		{8, 8},   // square: transpose path
		{16, 16}, // square, larger
		{5, 5},   // square, odd length
		{4, 6},   // non-square: strided path
		{6, 4},   // non-square, other orientation
		{1, 8},   // degenerate row dimension
		{8, 1},   // degenerate column dimension
	}

	for _, shape := range shapes {
		rows, cols := shape[0], shape[1]

		plan, err := NewPlanND[complex64]([]int{rows, cols})
		if err != nil {
			t.Fatalf("NewPlanND(%dx%d): %v", rows, cols, err)
		}

		src := randComplex64(rows*cols, int64(rows*100+cols))
		dst := make([]complex64, rows*cols)

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward(%dx%d): %v", rows, cols, err)
		}

		want := reference.NaiveDFT2D(src, rows, cols)
		for i := range want {
			if diff := cmplx.Abs(complex128(dst[i] - want[i])); diff > 1e-3 {
				t.Fatalf("%dx%d: dst[%d] = %v, want %v (diff %g)", rows, cols, i, dst[i], want[i], diff)
			}
		}
	}
}

func TestPlanND_3D_SquareTrailing_MatchesReference(t *testing.T) {
	shapes := [][3]int{
		{4, 8, 8}, // square trailing slab: transpose path per depth slice
		{3, 5, 5}, // odd sizes, square trailing
		{2, 4, 8}, // non-square trailing: strided path
		{5, 3, 4}, // all-different odd/even mix
	}

	for _, shape := range shapes {
		depth, height, width := shape[0], shape[1], shape[2]
		n := depth * height * width

		plan, err := NewPlanND[complex64]([]int{depth, height, width})
		if err != nil {
			t.Fatalf("NewPlanND(%dx%dx%d): %v", depth, height, width, err)
		}

		src := randComplex64(n, int64(n))
		dst := make([]complex64, n)

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward(%dx%dx%d): %v", depth, height, width, err)
		}

		want := reference.NaiveDFT3D(src, depth, height, width)
		for i := range want {
			if diff := cmplx.Abs(complex128(dst[i] - want[i])); diff > 1e-3 {
				t.Fatalf("%dx%dx%d: dst[%d] = %v, want %v (diff %g)",
					depth, height, width, i, dst[i], want[i], diff)
			}
		}
	}
}

func TestPlanND_2D_MatchesReference_Complex128(t *testing.T) {
	rows, cols := 12, 12 // square, non-power-of-two: transpose path

	plan, err := NewPlanND[complex128]([]int{rows, cols})
	if err != nil {
		t.Fatalf("NewPlanND: %v", err)
	}

	rng := rand.New(rand.NewSource(42))

	src := make([]complex128, rows*cols)
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	dst := make([]complex128, rows*cols)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := reference.NaiveDFT2D128(src, rows, cols)
	for i := range want {
		if diff := cmplx.Abs(dst[i] - want[i]); diff > 1e-10 {
			t.Fatalf("dst[%d] = %v, want %v (diff %g)", i, dst[i], want[i], diff)
		}
	}
}

// TestPlanND_RoundTrip_SquareTrailing round-trips shapes on the transpose
// path, including the in-place variant, to confirm forward and inverse take
// matching paths.
func TestPlanND_RoundTrip_SquareTrailing(t *testing.T) {
	for _, dims := range [][]int{{8, 8}, {3, 6, 6}, {2, 2, 4, 4}} {
		plan, err := NewPlanND[complex64](dims)
		if err != nil {
			t.Fatalf("NewPlanND(%v): %v", dims, err)
		}

		n := plan.Len()
		src := randComplex64(n, int64(n)+7)

		data := make([]complex64, n)
		copy(data, src)

		if err := plan.ForwardInPlace(data); err != nil {
			t.Fatalf("ForwardInPlace(%v): %v", dims, err)
		}

		if err := plan.InverseInPlace(data); err != nil {
			t.Fatalf("InverseInPlace(%v): %v", dims, err)
		}

		for i := range src {
			if diff := cmplx.Abs(complex128(data[i] - src[i])); diff > 1e-4 {
				t.Fatalf("%v: round-trip[%d] = %v, want %v (diff %g)", dims, i, data[i], src[i], diff)
			}
		}
	}
}
