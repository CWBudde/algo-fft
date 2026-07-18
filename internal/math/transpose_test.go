package math

import "testing"

func TestTransposeSquare(t *testing.T) {
	t.Parallel()

	// Cover sizes below, at, and straddling the tile edge, including
	// non-multiples of the block size.
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64, 100, 129} {
		data := make([]int, n*n)
		for i := range data {
			data[i] = i
		}

		TransposeSquare(data, n)

		for i := range n {
			for j := range n {
				got := data[i*n+j]

				want := j*n + i
				if got != want {
					t.Fatalf("n=%d: data[%d,%d] = %d, want %d", n, i, j, got, want)
				}
			}
		}
	}
}

// TestTransposeSquare_Involution verifies transposing twice restores the input.
func TestTransposeSquare_Involution(t *testing.T) {
	t.Parallel()

	n := 65

	data := make([]complex128, n*n)
	for i := range data {
		data[i] = complex(float64(i), float64(-i))
	}

	orig := make([]complex128, len(data))
	copy(orig, data)

	TransposeSquare(data, n)
	TransposeSquare(data, n)

	for i := range data {
		if data[i] != orig[i] {
			t.Fatalf("double transpose changed data[%d]: got %v, want %v", i, data[i], orig[i])
		}
	}
}

// TestTransposeSquareBlocked verifies every block size produces the same
// result as the reference walk, including block sizes larger than n.
func TestTransposeSquareBlocked(t *testing.T) {
	t.Parallel()

	for _, n := range []int{5, 16, 33, 96} {
		want := make([]int, n*n)
		for i := range n {
			for j := range n {
				want[i*n+j] = j*n + i
			}
		}

		for _, block := range []int{1, 2, 8, 16, 32, 64, 128} {
			data := make([]int, n*n)
			for i := range data {
				data[i] = i
			}

			transposeSquareBlocked(data, n, block)

			for i := range data {
				if data[i] != want[i] {
					t.Fatalf("n=%d block=%d: data[%d] = %d, want %d", n, block, i, data[i], want[i])
				}
			}
		}
	}
}

// TestTransposeSquare_DifferentTypes tests with different element types.
func TestTransposeSquare_DifferentTypes(t *testing.T) {
	t.Parallel()

	n := 3

	t.Run("float32", func(t *testing.T) {
		t.Parallel()

		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
		TransposeSquare(data, n)

		expected := []float32{1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0}
		for i := range data {
			if data[i] != expected[i] {
				t.Errorf("data[%d] = %v, want %v", i, data[i], expected[i])
			}
		}
	})

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		data := []complex64{1 + 0i, 2 + 0i, 3 + 0i, 4 + 0i, 5 + 0i, 6 + 0i, 7 + 0i, 8 + 0i, 9 + 0i}
		TransposeSquare(data, n)

		expected := []complex64{1 + 0i, 4 + 0i, 7 + 0i, 2 + 0i, 5 + 0i, 8 + 0i, 3 + 0i, 6 + 0i, 9 + 0i}
		for i := range data {
			if data[i] != expected[i] {
				t.Errorf("data[%d] = %v, want %v", i, data[i], expected[i])
			}
		}
	})
}
