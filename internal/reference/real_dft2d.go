package reference

import (
	"math"
	"math/cmplx"
)

// RealDFT2D computes the 2D DFT of a real-valued M×N matrix via naive O(M²N²) algorithm.
// This is used as a reference implementation for testing the optimized real FFT.
//
// Input: M×N row-major array of float32
// Output: M×(N/2+1) row-major array of complex64 (compact half-spectrum)
//
// Formula: X[k,l] = Σ(m=0..M-1) Σ(n=0..N-1) x[m,n] * exp(-2πi*(km/M + ln/N))
func RealDFT2D(input []float32, rows, cols int) []complex64 {
	if len(input) != rows*cols {
		panic("RealDFT2D: input length mismatch")
	}

	halfCols := cols/2 + 1
	output := make([]complex64, rows*halfCols)

	for k := 0; k < rows; k++ {
		for l := 0; l < halfCols; l++ {
			var sum complex128

			for m := 0; m < rows; m++ {
				for n := 0; n < cols; n++ {
					val := float64(input[m*cols+n])
					angle := -2 * math.Pi * (float64(k*m)/float64(rows) + float64(l*n)/float64(cols))
					twiddle := cmplx.Exp(complex(0, angle))
					sum += complex(val, 0) * twiddle
				}
			}

			output[k*halfCols+l] = complex64(sum)
		}
	}

	return output
}

// RealIDFT2D computes the 2D inverse DFT from a half-spectrum to real values.
//
// Input: M×(N/2+1) row-major array of complex64 (compact half-spectrum)
// Output: M×N row-major array of float32
//
// Formula: x[m,n] = (1/(M*N)) * Σ(k=0..M-1) Σ(l=0..N/2) X[k,l] * exp(2πi*(km/M + ln/N)) + conjugate terms
func RealIDFT2D(spectrum []complex64, rows, cols int) []float32 {
	if len(spectrum) != rows*(cols/2+1) {
		panic("RealIDFT2D: spectrum length mismatch")
	}

	halfCols := cols/2 + 1
	output := make([]float32, rows*cols)
	scale := 1.0 / float64(rows*cols)

	for m := 0; m < rows; m++ {
		for n := 0; n < cols; n++ {
			var sum complex128

			// Sum over positive frequencies (stored in spectrum)
			for k := 0; k < rows; k++ {
				for l := 0; l < halfCols; l++ {
					val := complex128(spectrum[k*halfCols+l])
					angle := 2 * math.Pi * (float64(k*m)/float64(rows) + float64(l*n)/float64(cols))
					twiddle := cmplx.Exp(complex(0, angle))
					sum += val * twiddle
				}
			}

			// Add conjugate terms for negative frequencies (l > N/2)
			for k := 0; k < rows; k++ {
				for l := halfCols; l < cols; l++ {
					// X[k, N-l] = conj(X[k, l])
					mirrorL := cols - l
					mirrorK := (rows - k) % rows
					val := complex128(spectrum[mirrorK*halfCols+mirrorL])
					valConj := complex(real(val), -imag(val))
					angle := 2 * math.Pi * (float64(k*m)/float64(rows) + float64(l*n)/float64(cols))
					twiddle := cmplx.Exp(complex(0, angle))
					sum += valConj * twiddle
				}
			}

			output[m*cols+n] = float32(real(sum) * scale)
		}
	}

	return output
}
