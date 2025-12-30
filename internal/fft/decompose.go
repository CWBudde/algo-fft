package fft

import (
	"sort"
)

// DecomposeStrategy describes how to split an FFT recursively.
type DecomposeStrategy struct {
	Size        int                 // Total FFT size
	SplitFactor int                 // Radix (2, 4, 8, or composite like 32)
	SubSize     int                 // Size of each sub-FFT
	NumSubs     int                 // Number of sub-FFTs (equal to SplitFactor)
	UseCodelet  bool                // True if this size has a codelet
	Recursive   *DecomposeStrategy  // Strategy for sub-problems (nil if codelet)
}

// PlanDecomposition finds the optimal split strategy for an FFT of size n.
// It recursively decomposes the problem until reaching sizes with codelets.
//
// Parameters:
//   - n: FFT size (must be power of 2)
//   - codeletSizes: Available codelet sizes (sorted ascending)
//   - cacheSize: L1 cache size in bytes for optimization
//
// Returns a decomposition strategy tree.
func PlanDecomposition(n int, codeletSizes []int, cacheSize int) *DecomposeStrategy {
	// Base case: n is a codelet size
	if hasCodelet(n, codeletSizes) {
		return &DecomposeStrategy{
			Size:       n,
			UseCodelet: true,
		}
	}

	// Special case: very small sizes (< smallest codelet) are treated as codelets
	// These will fall back to generic DIT implementation
	if len(codeletSizes) > 0 && n < codeletSizes[0] {
		return &DecomposeStrategy{
			Size:       n,
			UseCodelet: true, // Will use fallback generic DIT
		}
	}

	// Find all possible factorizations of n
	factors := findFactors(n)

	// Score each factorization based on cache fit, codelet availability,
	// radix size, and SIMD width
	bestScore := -1
	var bestStrategy *DecomposeStrategy

	for _, radix := range factors {
		subSize := n / radix

		score := scoreStrategy(radix, subSize, codeletSizes, cacheSize)
		if score > bestScore {
			bestScore = score
			bestStrategy = &DecomposeStrategy{
				Size:        n,
				SplitFactor: radix,
				SubSize:     subSize,
				NumSubs:     radix,
				UseCodelet:  false,
				Recursive:   PlanDecomposition(subSize, codeletSizes, cacheSize),
			}
		}
	}

	// Fallback: if no strategy found, use radix-2 split
	if bestStrategy == nil && len(factors) > 0 {
		radix := 2
		subSize := n / radix
		bestStrategy = &DecomposeStrategy{
			Size:        n,
			SplitFactor: radix,
			SubSize:     subSize,
			NumSubs:     radix,
			UseCodelet:  false,
			Recursive:   PlanDecomposition(subSize, codeletSizes, cacheSize),
		}
	}

	return bestStrategy
}

// scoreStrategy evaluates how good a particular radix split is.
// Higher scores are better.
func scoreStrategy(radix, subSize int, codeletSizes []int, cacheSize int) int {
	score := 0

	// HIGHEST PRIORITY: Prefer sub-problems that are codelets
	// This allows immediate use of optimized SIMD code
	if hasCodelet(subSize, codeletSizes) {
		score += 10000 // Much higher weight
	}

	// HIGH PRIORITY: Prefer sub-problems that fit in L1 cache
	// complex64 = 8 bytes, need 2 buffers (input + output)
	complexSize := subSize * 16
	if complexSize <= cacheSize {
		score += 500
	}

	// MEDIUM PRIORITY: Prefer larger radix (fewer stages)
	// But cap this to avoid choosing very large radix over codelet availability
	score += min(radix*10, 200)

	// MEDIUM PRIORITY: Prefer radix-4 for SIMD
	// AVX2 can process 4 complex64 values in parallel (256 bits / 64 bits)
	if radix == 4 {
		score += 100
	}

	// Prefer radix-8 for very large sizes
	if radix == 8 {
		score += 50
	}

	// LOW PRIORITY: Penalize very large radix (complex combine logic)
	// Beyond radix-8, the combine function becomes complicated
	if radix > 8 {
		score -= radix * 50 // Stronger penalty
	}

	return score
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// findFactors returns all divisors of n that are power-of-2, EXCLUDING n itself.
// Returns them in descending order (largest first).
//
// For example, findFactors(8192) returns [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
func findFactors(n int) []int {
	if !IsPowerOf2(n) {
		return []int{2} // Fallback to radix-2 for non-power-of-2
	}

	factors := []int{}
	// Start from 2, go up to n/2 (exclude n itself, since that would give subSize=1)
	for divisor := 2; divisor < n; divisor *= 2 {
		if n%divisor == 0 {
			factors = append(factors, divisor)
		}
	}

	// Return in descending order (prefer larger splits)
	sort.Sort(sort.Reverse(sort.IntSlice(factors)))
	return factors
}

// hasCodelet checks if a given size has a registered codelet.
func hasCodelet(size int, codeletSizes []int) bool {
	// Binary search since codeletSizes is sorted
	idx := sort.SearchInts(codeletSizes, size)
	return idx < len(codeletSizes) && codeletSizes[idx] == size
}

// DecompositionDepth calculates the maximum recursion depth of the strategy tree.
func (s *DecomposeStrategy) Depth() int {
	if s.UseCodelet || s.Recursive == nil {
		return 1
	}
	return 1 + s.Recursive.Depth()
}

// CodeletCount calculates the total number of codelet calls.
func (s *DecomposeStrategy) CodeletCount() int {
	if s.UseCodelet {
		return 1
	}
	if s.Recursive == nil {
		return 0
	}
	return s.NumSubs * s.Recursive.CodeletCount()
}

// String returns a human-readable description of the decomposition.
func (s *DecomposeStrategy) String() string {
	if s.UseCodelet {
		return "Codelet"
	}
	return "Split-" + string(rune('0'+s.SplitFactor))
}
