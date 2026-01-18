package transform

import "math"

// twiddle_recursive.go implements twiddle factor precomputation for recursive decomposition.

// TwiddleFactorsRecursive generates all twiddle factors needed for a decomposition strategy.
// This precomputes twiddles at plan creation time to ensure zero allocations during transforms.
func TwiddleFactorsRecursive[T Complex](strategy *DecomposeStrategy) []T {
	n := strategy.Size

	// Base case: codelet handles its own twiddles
	if strategy.UseCodelet {
		return ComputeTwiddleFactors[T](n)
	}

	// Calculate total twiddle size needed
	totalSize := estimateTwiddleSize(strategy)
	twiddles := make([]T, totalSize)

	// Fill twiddles recursively
	fillTwiddles(twiddles, strategy, 0)

	return twiddles
}

// estimateTwiddleSize calculates the total number of twiddle factors needed.
func estimateTwiddleSize(strategy *DecomposeStrategy) int {
	if strategy.UseCodelet || strategy.Recursive == nil {
		return strategy.Size // DIT codelets need N twiddle factors
	}

	// Current level needs: radix * subSize twiddles for combine step
	currentLevel := strategy.SplitFactor * strategy.SubSize

	// Plus recursive twiddles for all sub-problems
	subTwiddles := estimateTwiddleSize(strategy.Recursive)

	return currentLevel + strategy.NumSubs*subTwiddles
}

// fillTwiddles recursively fills the twiddle buffer.
func fillTwiddles[T Complex](buffer []T, strategy *DecomposeStrategy, offset int) int {
	n := strategy.Size

	// Base case: generate DIT-compatible twiddles
	if strategy.UseCodelet || strategy.Recursive == nil {
		twiddles := ComputeTwiddleFactors[T](n)
		copy(buffer[offset:], twiddles)

		return offset + len(twiddles)
	}

	radix := strategy.SplitFactor
	subSize := strategy.SubSize

	// Generate combine twiddles for this level
	// W_N^(r*k) for r = 0..radix-1, k = 0..subSize-1
	for r := range radix {
		for k := range subSize {
			angle := -2.0 * math.Pi * float64(r*k) / float64(n)
			buffer[offset] = makeComplexFromAngle[T](angle)
			offset++
		}
	}

	// Recursively fill sub-twiddles
	for range strategy.NumSubs {
		offset = fillTwiddles(buffer, strategy.Recursive, offset)
	}

	return offset
}

// generateTwiddleFactors generates standard FFT twiddle factors.
// Returns W_N^k for k = 0..N/2-1 where W_N = e^(-2πi/N).
func generateTwiddleFactors[T Complex](n int) []T {
	half := n / 2
	twiddles := make([]T, half)

	for k := range half {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		twiddles[k] = makeComplexFromAngle[T](angle)
	}

	return twiddles
}

// makeComplexFromAngle creates a complex number e^(iθ) = cos(θ) + i·sin(θ).
func makeComplexFromAngle[T Complex](angle float64) T {
	sin, cos := math.Sincos(angle)

	var zero T
	switch any(zero).(type) {
	case complex64:
		c := complex(float32(cos), float32(sin))
		rv, _ := any(c).(T)

		return rv
	case complex128:
		c := complex(cos, sin)
		rv, _ := any(c).(T)

		return rv
	default:
		panic("unsupported complex type")
	}
}
