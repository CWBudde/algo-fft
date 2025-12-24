package fft

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

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			got := data[i*n+j]
			want := j*n + i + 1
			if got != want {
				t.Fatalf("data[%d,%d] = %d, want %d", i, j, got, want)
			}
		}
	}
}
