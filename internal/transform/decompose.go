package transform

import (
	"sort"
)

// combineRadices lists the split factors PlanDecomposition may choose, largest
// first so the scorer's "prefer fewer stages" bias is evaluated in that order.
//
// Only radix 2 and 4 have a real butterfly (combineRadix2/combineRadix4).
// Every other radix is combined by evaluating a size-radix DFT directly —
// O(radix^2) complex multiplies per output element — which erases the point of
// decomposing at all. Bounding the radix is what keeps the strategy tree deep
// rather than wide: 16384 becomes 4x4096 -> 4x1024 -> 4x256-codelet instead of
// a single 32x512 split whose combine dominated the entire transform.
//
// Radix 8 is deliberately absent even though combineRadix8 exists: it is a
// direct 8-point DFT, not a butterfly, and splitting 8-way measured 34-44%
// slower than reaching the same size through two radix-4 levels (ABBA-
// interleaved, n=4096 and n=16384, both directions).
var combineRadices = [...]int{4, 2}

// DecomposeStrategy describes how to split an FFT recursively.
type DecomposeStrategy struct {
	Size        int                // Total FFT size
	SplitFactor int                // Radix chosen by the planner (see combineRadices)
	SubSize     int                // Size of each sub-FFT
	NumSubs     int                // Number of sub-FFTs (equal to SplitFactor)
	UseCodelet  bool               // True if this size has a codelet
	Recursive   *DecomposeStrategy // Strategy for sub-problems (nil if codelet)

	// LeafBitrev is the radix-2 bit-reversal permutation for a leaf of this
	// size, precomputed so the generic DIT fallback stays allocation-free.
	// It is set only on leaf nodes of power-of-two size, and is nil elsewhere
	// (ditForwardBitrev then computes the permutation itself).
	LeafBitrev []int
}

// newLeafStrategy builds a leaf node, precomputing the bit-reversal table the
// generic DIT fallback needs. Every leaf of a given tree has the same size, so
// this table is computed once per plan and shared by all leaf invocations.
func newLeafStrategy(n int) *DecomposeStrategy {
	leaf := &DecomposeStrategy{
		Size:       n,
		UseCodelet: true,
	}

	if IsPowerOf2(n) {
		leaf.LeafBitrev = ComputeBitReversalIndices(n)
	}

	return leaf
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
		return newLeafStrategy(n)
	}

	// Special case: very small sizes (< smallest codelet) are treated as codelets
	// These will fall back to generic DIT implementation
	if len(codeletSizes) > 0 && n < codeletSizes[0] {
		return newLeafStrategy(n)
	}

	// Score each candidate split based on cache fit, codelet availability,
	// radix size, and SIMD width. Only radices with a dedicated combine are
	// considered (see combineRadices).
	bestScore := -1

	var bestStrategy *DecomposeStrategy

	for _, radix := range combineRadices {
		if radix >= n || n%radix != 0 {
			continue
		}

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
	if bestStrategy == nil && n > 2 && n%2 == 0 {
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

// findFactors returns all divisors of n that are power-of-2, EXCLUDING n itself.
// Returns them in descending order (largest first).
//
// For example, findFactors(8192) returns [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2].
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
