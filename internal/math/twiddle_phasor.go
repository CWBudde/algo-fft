package math

import "math"

// Block sizes for phasor recursion.
// Chosen to keep accumulated floating-point error within acceptable bounds.
const (
	// PhasorBlockSize64 is the block size for complex64 phasor generation.
	// After 64 complex multiplications, error accumulates to ~7.7e-6.
	PhasorBlockSize64 = 64

	// PhasorBlockSize128 is the block size for complex128 phasor generation.
	// After 1024 complex multiplications, error accumulates to ~2.3e-13.
	PhasorBlockSize128 = 1024

	// PhasorThreshold is the minimum size to use phasor recursion.
	// Below this, direct computation is faster due to phasor setup overhead.
	PhasorThreshold = 32
)

// ComputeTwiddleFactorsPhasor generates twiddle factors using block-corrected
// phasor recursion with symmetric optimization. This is significantly faster
// than direct sin/cos computation for large n while maintaining accuracy.
//
// The algorithm:
//  1. Computes the step phasor W_1 = exp(-2πi/n) once
//  2. Uses complex multiplication W_k = W_{k-1} * W_1 within blocks
//  3. Re-anchors with exact sin/cos at block boundaries to reset error
//  4. Exploits conjugate symmetry: W_{n-k} = conj(W_k) to halve computation
//  5. Fixes special values exactly: W_0=1, W_{n/2}=-1, W_{n/4}=-i, W_{3n/4}=i
func ComputeTwiddleFactorsPhasor[T Complex](n int) []T {
	if n <= 0 {
		return nil
	}

	if n == 1 {
		return []T{ComplexFromFloat64[T](1, 0)}
	}

	// For small sizes, direct computation is faster
	if n < PhasorThreshold {
		return ComputeTwiddleFactors[T](n)
	}

	twiddle := make([]T, n)
	blockSize := selectBlockSize[T]()

	// W_0 = 1 (exact)
	twiddle[0] = ComplexFromFloat64[T](1, 0)

	// Compute the step phasor W_1 = exp(-2πi/n)
	stepAngle := -2.0 * math.Pi / float64(n)
	stepSin, stepCos := math.Sincos(stepAngle)
	stepRe, stepIm := stepCos, stepSin

	// Only compute first half+1, derive rest from symmetry
	halfN := n / 2

	// Generate twiddles using block-corrected phasor recursion
	k := 1
	for k <= halfN {
		blockEnd := min(k+blockSize-1, halfN)

		// Get the starting value for this block
		var prevRe, prevIm float64
		if k == 1 {
			// First block starts from W_1 computed directly
			prevRe, prevIm = stepCos, stepSin
			twiddle[1] = ComplexFromFloat64[T](prevRe, prevIm)
			k = 2
		} else {
			// Re-anchor at block start with exact computation
			anchorAngle := -2.0 * math.Pi * float64(k) / float64(n)
			anchorSin, anchorCos := math.Sincos(anchorAngle)
			prevRe, prevIm = anchorCos, anchorSin
			twiddle[k] = ComplexFromFloat64[T](prevRe, prevIm)
			k++
		}

		// Use phasor recursion within the block
		for k <= blockEnd {
			// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
			newRe := prevRe*stepRe - prevIm*stepIm
			newIm := prevRe*stepIm + prevIm*stepRe
			twiddle[k] = ComplexFromFloat64[T](newRe, newIm)
			prevRe, prevIm = newRe, newIm
			k++
		}
	}

	// Apply conjugate symmetry: W_{n-k} = conj(W_k)
	for k := 1; k < halfN; k++ {
		twiddle[n-k] = Conj(twiddle[k])
	}

	// Fix special values exactly (override any accumulated error)
	fixSpecialValues(twiddle, n)

	return twiddle
}

// selectBlockSize returns the appropriate block size based on the complex type.
func selectBlockSize[T Complex]() int {
	var zero T
	switch any(zero).(type) {
	case complex64:
		return PhasorBlockSize64
	case complex128:
		return PhasorBlockSize128
	default:
		return PhasorBlockSize64
	}
}

// fixSpecialValues sets mathematically exact values for special indices.
// These values are exactly representable in floating-point.
func fixSpecialValues[T Complex](twiddle []T, n int) {
	// W_0 = 1 (already set, but ensure exactness)
	twiddle[0] = ComplexFromFloat64[T](1, 0)

	// W_{n/2} = -1 (for even n)
	if n >= 2 && n%2 == 0 {
		twiddle[n/2] = ComplexFromFloat64[T](-1, 0)
	}

	// W_{n/4} = -i (for n divisible by 4)
	if n >= 4 && n%4 == 0 {
		twiddle[n/4] = ComplexFromFloat64[T](0, -1)
	}

	// W_{3n/4} = i (for n divisible by 4)
	if n >= 4 && n%4 == 0 {
		twiddle[3*n/4] = ComplexFromFloat64[T](0, 1)
	}
}
