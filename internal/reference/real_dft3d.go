package reference

import (
	"math"
	"math/cmplx"
)

// RealDFT3D computes the 3D DFT of a real-valued D×H×W volume via naive O(D²H²W²) algorithm.
// This is used as a reference implementation for testing the optimized real FFT.
//
// Input: D×H×W row-major array of float32
// Output: D×H×(W/2+1) row-major array of complex64 (compact half-spectrum)
//
// Formula: X[kd,kh,kw] = Σ(d=0..D-1) Σ(h=0..H-1) Σ(w=0..W-1)
//
//	x[d,h,w] * exp(-2πi*(kd*d/D + kh*h/H + kw*w/W))
func RealDFT3D(input []float32, depth, height, width int) []complex64 {
	if len(input) != depth*height*width {
		panic("RealDFT3D: input length mismatch")
	}

	halfWidth := width/2 + 1
	output := make([]complex64, depth*height*halfWidth)

	for kd := 0; kd < depth; kd++ {
		for kh := 0; kh < height; kh++ {
			for kw := 0; kw < halfWidth; kw++ {
				var sum complex128

				for d := 0; d < depth; d++ {
					for h := 0; h < height; h++ {
						for w := 0; w < width; w++ {
							val := float64(input[d*height*width+h*width+w])
							angle := -2 * math.Pi * (
								float64(kd*d)/float64(depth) +
									float64(kh*h)/float64(height) +
									float64(kw*w)/float64(width))
							twiddle := cmplx.Exp(complex(0, angle))
							sum += complex(val, 0) * twiddle
						}
					}
				}

				output[kd*height*halfWidth+kh*halfWidth+kw] = complex64(sum)
			}
		}
	}

	return output
}

// RealIDFT3D computes the 3D inverse DFT from a half-spectrum to real values.
//
// Input: D×H×(W/2+1) row-major array of complex64 (compact half-spectrum)
// Output: D×H×W row-major array of float32
//
// Formula: x[d,h,w] = (1/(D*H*W)) * Σ X[kd,kh,kw] * exp(2πi*(...)) + conjugate terms
func RealIDFT3D(spectrum []complex64, depth, height, width int) []float32 {
	if len(spectrum) != depth*height*(width/2+1) {
		panic("RealIDFT3D: spectrum length mismatch")
	}

	halfWidth := width/2 + 1
	output := make([]float32, depth*height*width)
	scale := 1.0 / float64(depth*height*width)

	for d := 0; d < depth; d++ {
		for h := 0; h < height; h++ {
			for w := 0; w < width; w++ {
				var sum complex128

				// Sum over positive frequencies (stored in spectrum)
				for kd := 0; kd < depth; kd++ {
					for kh := 0; kh < height; kh++ {
						for kw := 0; kw < halfWidth; kw++ {
							val := complex128(spectrum[kd*height*halfWidth+kh*halfWidth+kw])
							angle := 2 * math.Pi * (
								float64(kd*d)/float64(depth) +
									float64(kh*h)/float64(height) +
									float64(kw*w)/float64(width))
							twiddle := cmplx.Exp(complex(0, angle))
							sum += val * twiddle
						}
					}
				}

				// Add conjugate terms for negative frequencies (kw > W/2)
				for kd := 0; kd < depth; kd++ {
					for kh := 0; kh < height; kh++ {
						for kw := halfWidth; kw < width; kw++ {
							// X[kd, kh, W-kw] = conj(X[-kd, -kh, kw])
							mirrorKW := width - kw
							mirrorKD := (depth - kd) % depth
							mirrorKH := (height - kh) % height
							val := complex128(spectrum[mirrorKD*height*halfWidth+mirrorKH*halfWidth+mirrorKW])
							valConj := complex(real(val), -imag(val))
							angle := 2 * math.Pi * (
								float64(kd*d)/float64(depth) +
									float64(kh*h)/float64(height) +
									float64(kw*w)/float64(width))
							twiddle := cmplx.Exp(complex(0, angle))
							sum += valConj * twiddle
						}
					}
				}

				output[d*height*width+h*width+w] = float32(real(sum) * scale)
			}
		}
	}

	return output
}
