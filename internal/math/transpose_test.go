package math

import "testing"

func TestComputeSquareTransposePairs(t *testing.T) {
	t.Parallel()

	n := 4

	pairs := ComputeSquareTransposePairs(n)
	if len(pairs) != n*(n-1)/2 {
		t.Fatalf("pairs length = %d, want %d", len(pairs), n*(n-1)/2)
	}

	data := make([]int, n*n)
	for i := range data {
		data[i] = i + 1
	}

	ApplyTransposePairs(data, pairs)

	for i := range n {
		for j := range n {
			got := data[i*n+j]

			want := j*n + i + 1
			if got != want {
				t.Fatalf("data[%d,%d] = %d, want %d", i, j, got, want)
			}
		}
	}
}

// TestComputeSquareTransposePairs_EdgeCases tests edge cases.
func TestComputeSquareTransposePairs_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("zero", func(t *testing.T) {
		t.Parallel()

		pairs := ComputeSquareTransposePairs(0)
		if pairs != nil {
			t.Errorf("ComputeSquareTransposePairs(0) should return nil, got %v", pairs)
		}
	})

	t.Run("negative", func(t *testing.T) {
		t.Parallel()

		pairs := ComputeSquareTransposePairs(-1)
		if pairs != nil {
			t.Errorf("ComputeSquareTransposePairs(-1) should return nil, got %v", pairs)
		}
	})

	t.Run("n=1", func(t *testing.T) {
		t.Parallel()

		pairs := ComputeSquareTransposePairs(1)
		if len(pairs) != 0 {
			t.Errorf("ComputeSquareTransposePairs(1) should return empty slice, got %v", pairs)
		}
	})
}

// TestComputeSquareTransposePairs_Caching tests that caching works.
func TestComputeSquareTransposePairs_Caching(t *testing.T) {
	t.Parallel()

	n := 8

	// First call - should cache
	pairs1 := ComputeSquareTransposePairs(n)

	// Second call - should return cached
	pairs2 := ComputeSquareTransposePairs(n)

	// Compare lengths
	if len(pairs1) != len(pairs2) {
		t.Fatalf("cached pairs length mismatch: %d vs %d", len(pairs1), len(pairs2))
	}

	// Verify they're the same
	for i := range pairs1 {
		if pairs1[i] != pairs2[i] {
			t.Errorf("pairs mismatch at index %d: %v vs %v", i, pairs1[i], pairs2[i])
		}
	}
}

// TestApplyTransposePairs_DifferentTypes tests with different types.
func TestApplyTransposePairs_DifferentTypes(t *testing.T) {
	t.Parallel()

	n := 3
	pairs := ComputeSquareTransposePairs(n)

	t.Run("float32", func(t *testing.T) {
		t.Parallel()

		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
		ApplyTransposePairs(data, pairs)

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
		ApplyTransposePairs(data, pairs)

		expected := []complex64{1 + 0i, 4 + 0i, 7 + 0i, 2 + 0i, 5 + 0i, 8 + 0i, 3 + 0i, 6 + 0i, 9 + 0i}
		for i := range data {
			if data[i] != expected[i] {
				t.Errorf("data[%d] = %v, want %v", i, data[i], expected[i])
			}
		}
	})
}
