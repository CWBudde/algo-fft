package fft

import (
	"testing"
)

// TestTwiddleFactorComparison - Verify twiddle factor generation conventions.
// ComputeTwiddleFactors generates N factors (used by DIT codelets),
// while generateTwiddleFactors generates N/2 factors (legacy convention).
// TwiddleFactorsRecursive correctly uses ComputeTwiddleFactors for DIT compatibility.
func TestTwiddleFactorComparison(t *testing.T) {
	t.Parallel()

	size := 512

	// Method 1: ComputeTwiddleFactors (what DIT expects)
	twiddle1 := ComputeTwiddleFactors[complex64](size)

	// Method 2: generateTwiddleFactors (legacy, generates N/2 factors)
	twiddle2 := generateTwiddleFactors[complex64](size)

	t.Logf("ComputeTwiddleFactors returned %d factors (for DIT codelets)", len(twiddle1))
	t.Logf("generateTwiddleFactors returned %d factors (legacy convention)", len(twiddle2))

	// Verify conventions
	if len(twiddle1) != size {
		t.Errorf("ComputeTwiddleFactors: expected %d factors, got %d", size, len(twiddle1))
	}

	if len(twiddle2) != size/2 {
		t.Errorf("generateTwiddleFactors: expected %d factors, got %d", size/2, len(twiddle2))
	}

	// Expected difference - this is correct behavior, not a bug
	if len(twiddle1) == len(twiddle2) {
		t.Errorf("Expected twiddle factor length difference: ComputeTwiddleFactors uses N, generateTwiddleFactors uses N/2")
	}
}
