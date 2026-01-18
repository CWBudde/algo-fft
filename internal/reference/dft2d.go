// Package reference provides naive O(n²) DFT implementations for validation.
package reference

import (
	"math"
	"math/cmplx"
)

// NaiveDFT2D computes the 2D Discrete Fourier Transform using the naive O(M²N²) algorithm.
// Input is row-major: src[row*cols + col].
// Output is frequency domain in row-major: X[k*cols + l] where k is row frequency, l is column frequency.
//
// Formula: X[k,l] = Σ(m=0..rows-1) Σ(n=0..cols-1) x[m,n] * exp(-2πi*(km/rows + ln/cols))
//
// This function is intended for testing correctness of optimized implementations.
// For production use, use the optimized Plan2D from the algo-fft package.
func NaiveDFT2D(src []complex64, rows, cols int) []complex64 {
	if len(src) != rows*cols {
		panic("dft2d: input length must equal rows*cols")
	}

	// Allocate output
	dst := make([]complex64, rows*cols)

	// For each output frequency bin (k, l)
	for k := range rows {
		for l := range cols {
			var sum complex128 // Use higher precision for accumulation

			// Sum over all input samples (m, n)
			for m := range rows {
				for n := range cols {
					// Compute phase: -2π*(km/rows + ln/cols)
					phaseRow := -2.0 * math.Pi * float64(k*m) / float64(rows)
					phaseCol := -2.0 * math.Pi * float64(l*n) / float64(cols)
					phase := phaseRow + phaseCol

					// exp(i*phase) = cos(phase) + i*sin(phase)
					twiddle := complex(math.Cos(phase), math.Sin(phase))

					// Accumulate: sum += x[m,n] * twiddle
					idx := m*cols + n
					sum += complex128(src[idx]) * twiddle
				}
			}

			// Store result
			dst[k*cols+l] = complex64(sum)
		}
	}

	return dst
}

// NaiveDFT2D128 is the complex128 version of NaiveDFT2D.
// It provides higher precision for validation of complex128 transforms.
func NaiveDFT2D128(src []complex128, rows, cols int) []complex128 {
	if len(src) != rows*cols {
		panic("dft2d: input length must equal rows*cols")
	}

	dst := make([]complex128, rows*cols)

	for row := range rows {
		for col := range cols {
			var sum complex128

			for m := range rows {
				for n := range cols {
					phaseRow := -2.0 * math.Pi * float64(row*m) / float64(rows)
					phaseCol := -2.0 * math.Pi * float64(col*n) / float64(cols)
					phase := phaseRow + phaseCol

					twiddle := complex(math.Cos(phase), math.Sin(phase))

					idx := m*cols + n
					sum += src[idx] * twiddle
				}
			}

			dst[row*cols+col] = sum
		}
	}

	return dst
}

// NaiveIDFT2D computes the 2D Inverse Discrete Fourier Transform using the naive O(M²N²) algorithm.
//
// Formula: x[m,n] = (1/(rows*cols)) * Σ(k=0..rows-1) Σ(l=0..cols-1) X[k,l] * exp(2πi*(km/rows + ln/cols))
//
// Note the normalization factor 1/(rows*cols) and the positive phase (inverse uses +2πi).
func NaiveIDFT2D(src []complex64, rows, cols int) []complex64 {
	if len(src) != rows*cols {
		panic("dft2d: input length must equal rows*cols")
	}

	dst := make([]complex64, rows*cols)
	scale := 1.0 / float64(rows*cols) // Normalization factor

	for m := range rows {
		for n := range cols {
			var sum complex128

			for row := range rows {
				for col := range cols {
					// Positive phase for inverse: +2π*(km/rows + ln/cols)
					phaseRow := 2.0 * math.Pi * float64(row*m) / float64(rows)
					phaseCol := 2.0 * math.Pi * float64(col*n) / float64(cols)
					phase := phaseRow + phaseCol

					twiddle := complex(math.Cos(phase), math.Sin(phase))

					idx := row*cols + col
					sum += complex128(src[idx]) * twiddle
				}
			}

			// Apply normalization
			dst[m*cols+n] = complex64(sum * complex(scale, 0))
		}
	}

	return dst
}

// NaiveIDFT2D128 is the complex128 version of NaiveIDFT2D.
func NaiveIDFT2D128(src []complex128, rows, cols int) []complex128 {
	if len(src) != rows*cols {
		panic("dft2d: input length must equal rows*cols")
	}

	dst := make([]complex128, rows*cols)
	scale := 1.0 / float64(rows*cols)

	for m := range rows {
		for n := range cols {
			var sum complex128

			for k := range rows {
				for l := range cols {
					phaseRow := 2.0 * math.Pi * float64(k*m) / float64(rows)
					phaseCol := 2.0 * math.Pi * float64(l*n) / float64(cols)
					phase := phaseRow + phaseCol

					twiddle := complex(math.Cos(phase), math.Sin(phase))

					idx := k*cols + l
					sum += src[idx] * twiddle
				}
			}

			dst[m*cols+n] = sum * complex(scale, 0)
		}
	}

	return dst
}

// absComplex64 returns the magnitude of a complex64 number.
func absComplex64(v complex64) float64 {
	return cmplx.Abs(complex128(v))
}

// absComplex128 returns the magnitude of a complex128 number.
func absComplex128(v complex128) float64 {
	return cmplx.Abs(v)
}
