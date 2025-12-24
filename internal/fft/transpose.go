package fft

// TransposePair describes a swap between two indices in a flattened matrix.
type TransposePair struct {
	I int
	J int
}

// ComputeSquareTransposePairs returns swap pairs to transpose an n x n matrix
// stored in row-major order. n must be positive.
func ComputeSquareTransposePairs(n int) []TransposePair {
	if n <= 0 {
		return nil
	}

	pairs := make([]TransposePair, 0, n*(n-1)/2)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a := i*n + j
			b := j*n + i
			pairs = append(pairs, TransposePair{I: a, J: b})
		}
	}

	return pairs
}

// ApplyTransposePairs swaps elements in-place using the provided pairs.
// The caller is responsible for ensuring the pairs match the matrix layout.
func ApplyTransposePairs[T any](data []T, pairs []TransposePair) {
	for _, pair := range pairs {
		data[pair.I], data[pair.J] = data[pair.J], data[pair.I]
	}
}
