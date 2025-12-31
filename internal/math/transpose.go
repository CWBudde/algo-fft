package math

import "sync"

//nolint:gochecknoglobals
var transposeCache struct {
	sync.RWMutex

	pairs map[int][]TransposePair
}

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

	transposeCache.RLock()

	if transposeCache.pairs != nil {
		if cached, ok := transposeCache.pairs[n]; ok {
			transposeCache.RUnlock()
			return cached
		}
	}

	transposeCache.RUnlock()

	pairs := make([]TransposePair, 0, n*(n-1)/2)
	for i := range n {
		for j := i + 1; j < n; j++ {
			a := i*n + j
			b := j*n + i
			pairs = append(pairs, TransposePair{I: a, J: b})
		}
	}

	transposeCache.Lock()

	if transposeCache.pairs == nil {
		transposeCache.pairs = make(map[int][]TransposePair)
	}

	transposeCache.pairs[n] = pairs
	transposeCache.Unlock()

	return pairs
}

// ApplyTransposePairs swaps elements in-place using the provided pairs.
// The caller is responsible for ensuring the pairs match the matrix layout.
func ApplyTransposePairs[T any](data []T, pairs []TransposePair) {
	for _, pair := range pairs {
		data[pair.I], data[pair.J] = data[pair.J], data[pair.I]
	}
}
