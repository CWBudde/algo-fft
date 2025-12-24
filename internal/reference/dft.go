// Package reference provides naive O(n²) DFT implementations for testing.
// These implementations prioritize correctness over performance and use
// complex128 internally for higher precision reference values.
package reference

import "math"

// NaiveDFT computes the Discrete Fourier Transform using the direct O(n²) formula.
// It uses complex128 arithmetic internally and converts back to complex64 for the result.
//
// The forward DFT is defined as:
//
//	X[k] = Σ(n=0 to N-1) x[n] * exp(-2πi*k*n/N)
//
// where k = 0, 1, ..., N-1.
func NaiveDFT(src []complex64) []complex64 {
	n := len(src)
	if n == 0 {
		return nil
	}

	// Convert input to complex128 for higher precision
	input := make([]complex128, n)
	for i, v := range src {
		input[i] = complex128(v)
	}

	// Compute DFT using the direct formula
	output := make([]complex128, n)
	for k := range n {
		var sum complex128
		for m := range n {
			// W_n^(k*m) = exp(-2πi*k*m/N)
			angle := -2.0 * math.Pi * float64(k) * float64(m) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += input[m] * twiddle
		}
		output[k] = sum
	}

	// Convert back to complex64
	result := make([]complex64, n)
	for i, v := range output {
		result[i] = complex64(v)
	}

	return result
}

// NaiveIDFT computes the Inverse Discrete Fourier Transform using the direct O(n²) formula.
// It uses complex128 arithmetic internally and converts back to complex64 for the result.
//
// The inverse DFT is defined as:
//
//	x[n] = (1/N) * Σ(k=0 to N-1) X[k] * exp(2πi*k*n/N)
//
// where n = 0, 1, ..., N-1.
func NaiveIDFT(src []complex64) []complex64 {
	n := len(src)
	if n == 0 {
		return nil
	}

	// Convert input to complex128 for higher precision
	input := make([]complex128, n)
	for i, v := range src {
		input[i] = complex128(v)
	}

	// Compute IDFT using the direct formula
	output := make([]complex128, n)
	scale := 1.0 / float64(n)
	for m := range n {
		var sum complex128
		for k := range n {
			// W_n^(-k*m) = exp(2πi*k*m/N) (positive exponent for inverse)
			angle := 2.0 * math.Pi * float64(k) * float64(m) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += input[k] * twiddle
		}
		output[m] = sum * complex(scale, 0)
	}

	// Convert back to complex64
	result := make([]complex64, n)
	for i, v := range output {
		result[i] = complex64(v)
	}

	return result
}

// NaiveDFT128 computes the Discrete Fourier Transform using complex128 throughout.
// This is useful when maximum precision is needed for reference comparisons.
func NaiveDFT128(src []complex128) []complex128 {
	n := len(src)
	if n == 0 {
		return nil
	}

	output := make([]complex128, n)
	for k := range n {
		var sum complex128
		for m := range n {
			angle := -2.0 * math.Pi * float64(k) * float64(m) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += src[m] * twiddle
		}
		output[k] = sum
	}

	return output
}

// NaiveIDFT128 computes the Inverse Discrete Fourier Transform using complex128 throughout.
// This is useful when maximum precision is needed for reference comparisons.
func NaiveIDFT128(src []complex128) []complex128 {
	n := len(src)
	if n == 0 {
		return nil
	}

	output := make([]complex128, n)
	scale := 1.0 / float64(n)
	for m := range n {
		var sum complex128
		for k := range n {
			angle := 2.0 * math.Pi * float64(k) * float64(m) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += src[k] * twiddle
		}
		output[m] = sum * complex(scale, 0)
	}

	return output
}
