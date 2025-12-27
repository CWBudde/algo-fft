// Package reference provides naive O(n²) DFT implementations for validation.
package reference

import (
	"math"
)

// NaiveDFT3D computes the 3D Discrete Fourier Transform using the naive O(D³H³W³) algorithm.
// Input is in row-major order: src[d*height*width + h*width + w] where d is depth, h is height, w is width.
// Output is frequency domain in row-major: X[kd*height*width + kh*width + kw].
//
// Formula: X[kd,kh,kw] = Σ(d=0..depth-1) Σ(h=0..height-1) Σ(w=0..width-1)
//
//	x[d,h,w] * exp(-2πi*(kd*d/depth + kh*h/height + kw*w/width))
//
// This function is intended for testing correctness of optimized implementations.
// For production use, use the optimized Plan3D from the algo-fft package.
func NaiveDFT3D(src []complex64, depth, height, width int) []complex64 {
	if len(src) != depth*height*width {
		panic("dft3d: input length must equal depth*height*width")
	}

	// Allocate output
	dst := make([]complex64, depth*height*width)

	// For each output frequency bin (kd, kh, kw)
	for kd := range depth {
		for kh := range height {
			for kw := range width {
				var sum complex128 // Use higher precision for accumulation

				// Sum over all input samples (d, h, w)
				for d := range depth {
					for h := range height {
						for w := range width {
							// Compute phase: -2π*(kd*d/depth + kh*h/height + kw*w/width)
							phaseDepth := -2.0 * math.Pi * float64(kd*d) / float64(depth)
							phaseHeight := -2.0 * math.Pi * float64(kh*h) / float64(height)
							phaseWidth := -2.0 * math.Pi * float64(kw*w) / float64(width)
							phase := phaseDepth + phaseHeight + phaseWidth

							// exp(i*phase) = cos(phase) + i*sin(phase)
							twiddle := complex(math.Cos(phase), math.Sin(phase))

							// Accumulate: sum += x[d,h,w] * twiddle
							idx := d*height*width + h*width + w
							sum += complex128(src[idx]) * twiddle
						}
					}
				}

				// Store result
				dst[kd*height*width+kh*width+kw] = complex64(sum)
			}
		}
	}

	return dst
}

// NaiveDFT3D128 is the complex128 version of NaiveDFT3D.
// It provides higher precision for validation of complex128 transforms.
func NaiveDFT3D128(src []complex128, depth, height, width int) []complex128 {
	if len(src) != depth*height*width {
		panic("dft3d: input length must equal depth*height*width")
	}

	dst := make([]complex128, depth*height*width)

	for kd := range depth {
		for kh := range height {
			for kw := range width {
				var sum complex128

				for d := range depth {
					for h := range height {
						for w := range width {
							phaseDepth := -2.0 * math.Pi * float64(kd*d) / float64(depth)
							phaseHeight := -2.0 * math.Pi * float64(kh*h) / float64(height)
							phaseWidth := -2.0 * math.Pi * float64(kw*w) / float64(width)
							phase := phaseDepth + phaseHeight + phaseWidth

							twiddle := complex(math.Cos(phase), math.Sin(phase))

							idx := d*height*width + h*width + w
							sum += src[idx] * twiddle
						}
					}
				}

				dst[kd*height*width+kh*width+kw] = sum
			}
		}
	}

	return dst
}

// NaiveIDFT3D computes the 3D Inverse Discrete Fourier Transform using the naive O(D³H³W³) algorithm.
//
// Formula: x[d,h,w] = (1/(depth*height*width)) * Σ(kd=0..depth-1) Σ(kh=0..height-1) Σ(kw=0..width-1)
//
//	X[kd,kh,kw] * exp(2πi*(kd*d/depth + kh*h/height + kw*w/width))
//
// Note the normalization factor 1/(depth*height*width) and the positive phase (inverse uses +2πi).
func NaiveIDFT3D(src []complex64, depth, height, width int) []complex64 {
	if len(src) != depth*height*width {
		panic("dft3d: input length must equal depth*height*width")
	}

	dst := make([]complex64, depth*height*width)
	scale := 1.0 / float64(depth*height*width) // Normalization factor

	for d := range depth {
		for h := range height {
			for w := range width {
				var sum complex128

				for kd := range depth {
					for kh := range height {
						for kw := range width {
							// Positive phase for inverse: +2π*(kd*d/depth + kh*h/height + kw*w/width)
							phaseDepth := 2.0 * math.Pi * float64(kd*d) / float64(depth)
							phaseHeight := 2.0 * math.Pi * float64(kh*h) / float64(height)
							phaseWidth := 2.0 * math.Pi * float64(kw*w) / float64(width)
							phase := phaseDepth + phaseHeight + phaseWidth

							twiddle := complex(math.Cos(phase), math.Sin(phase))

							idx := kd*height*width + kh*width + kw
							sum += complex128(src[idx]) * twiddle
						}
					}
				}

				// Apply normalization
				dst[d*height*width+h*width+w] = complex64(sum * complex(scale, 0))
			}
		}
	}

	return dst
}

// NaiveIDFT3D128 is the complex128 version of NaiveIDFT3D.
func NaiveIDFT3D128(src []complex128, depth, height, width int) []complex128 {
	if len(src) != depth*height*width {
		panic("dft3d: input length must equal depth*height*width")
	}

	dst := make([]complex128, depth*height*width)
	scale := 1.0 / float64(depth*height*width)

	for d := range depth {
		for h := range height {
			for w := range width {
				var sum complex128

				for kd := range depth {
					for kh := range height {
						for kw := range width {
							phaseDepth := 2.0 * math.Pi * float64(kd*d) / float64(depth)
							phaseHeight := 2.0 * math.Pi * float64(kh*h) / float64(height)
							phaseWidth := 2.0 * math.Pi * float64(kw*w) / float64(width)
							phase := phaseDepth + phaseHeight + phaseWidth

							twiddle := complex(math.Cos(phase), math.Sin(phase))

							idx := kd*height*width + kh*width + kw
							sum += src[idx] * twiddle
						}
					}
				}

				dst[d*height*width+h*width+w] = sum * complex(scale, 0)
			}
		}
	}

	return dst
}
