// Package reference provides naive O(n²) DFT implementations for testing.
// These implementations prioritize correctness over performance and use
// complex128 internally for higher precision reference values.
package reference

import "math"

// NaiveDFTWide computes the Discrete Fourier Transform of a complex64 input
// using the direct O(n²) formula and returns the complex128 accumulator
// without narrowing it back to complex64.
//
// This is the float64 reference for a complex64 transform: the difference
// between a complex64 FFT and NaiveDFTWide of the same input measures that FFT's
// error and essentially nothing else. NaiveDFT narrows these same values to
// complex64 and therefore carries a comparable error of its own; it is a
// correctness oracle, not an accuracy reference.
//
// Its own relative error grows roughly as n·2⁻⁵³, not as a constant, because the
// twiddle angle -2πkm/n is formed without reduction mod 2π and so reaches a
// magnitude of ~2πn, where a float64's rounding is proportionally coarser.
// Measured: 8.6e-16 at n = 8 rising to 6.7e-13 at n = 4096, doubling with each
// doubling of n. That stays several orders of magnitude below float32 epsilon
// (1.19e-07) until about n = 10⁸, so this is a sound reference at any size worth
// running — but it is not a valid reference for a complex128 transform, whose own
// error is far smaller than the reference's above n ≈ 16.
//
// Pass the exact float32 vector the transform received. Referencing an
// unrounded float64 draw instead would fold input quantization — an error the
// transform never committed — into the measurement, inflating it by roughly
// float32 epsilon.
//
// The forward DFT is defined as:
//
//	X[k] = Σ(n=0 to N-1) x[n] * exp(-2πi*k*n/N)
//
// where k = 0, 1, ..., N-1.
func NaiveDFTWide(src []complex64) []complex128 {
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

	for freqBin := range n {
		var sum complex128

		for sampleIdx := range n {
			// W_n^(k*m) = exp(-2πi*k*m/N)
			angle := -2.0 * math.Pi * float64(freqBin) * float64(sampleIdx) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += input[sampleIdx] * twiddle
		}

		output[freqBin] = sum
	}

	return output
}

// NaiveDFT computes the Discrete Fourier Transform using the direct O(n²) formula.
// It uses complex128 arithmetic internally and converts back to complex64 for the result.
//
// The narrowing costs about as much accuracy as the transform under test, so
// this is a correctness oracle rather than an accuracy reference — use
// NaiveDFTWide to measure a complex64 transform's error.
//
// The forward DFT is defined as:
//
//	X[k] = Σ(n=0 to N-1) x[n] * exp(-2πi*k*n/N)
//
// where k = 0, 1, ..., N-1.
func NaiveDFT(src []complex64) []complex64 {
	output := NaiveDFTWide(src)
	if output == nil {
		return nil
	}

	// Convert back to complex64
	result := make([]complex64, len(output))
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

	for sampleIdx := range n {
		var sum complex128

		for freqBin := range n {
			// W_n^(-k*m) = exp(2πi*k*m/N) (positive exponent for inverse)
			angle := 2.0 * math.Pi * float64(freqBin) * float64(sampleIdx) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += input[freqBin] * twiddle
		}

		output[sampleIdx] = sum * complex(scale, 0)
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

	for freqBin := range n {
		var sum complex128

		for sampleIdx := range n {
			angle := -2.0 * math.Pi * float64(freqBin) * float64(sampleIdx) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += src[sampleIdx] * twiddle
		}

		output[freqBin] = sum
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

	for sampleIdx := range n {
		var sum complex128

		for freqBin := range n {
			angle := 2.0 * math.Pi * float64(freqBin) * float64(sampleIdx) / float64(n)
			twiddle := complex(math.Cos(angle), math.Sin(angle))
			sum += src[freqBin] * twiddle
		}

		output[sampleIdx] = sum * complex(scale, 0)
	}

	return output
}
