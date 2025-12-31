package fft

import "github.com/MeKo-Christian/algo-fft/internal/math"

// Re-export transpose types and functions from internal/math for backward compatibility.

// TransposePair describes a swap between two indices in a flattened matrix.
type TransposePair = math.TransposePair

// ComputeSquareTransposePairs returns swap pairs to transpose an n x n matrix
// stored in row-major order. n must be positive.
func ComputeSquareTransposePairs(n int) []TransposePair {
	return math.ComputeSquareTransposePairs(n)
}

// ApplyTransposePairs swaps elements in-place using the provided pairs.
// The caller is responsible for ensuring the pairs match the matrix layout.
func ApplyTransposePairs[T any](data []T, pairs []TransposePair) {
	math.ApplyTransposePairs(data, pairs)
}
