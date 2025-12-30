package fft

import "math"

// combine.go implements the "combine" step of Cooley-Tukey decomposition.
// After computing sub-FFTs, these functions merge them using twiddle factors.

// combineRadix2 combines two N/2-point FFTs into an N-point FFT.
// This is the classic Cooley-Tukey radix-2 decimation-in-time combine step.
//
// Algorithm:
//   For k = 0 to N/2-1:
//     t = twiddle[k] * sub1[k]     // Twiddle multiplication
//     dst[k]     = sub0[k] + t      // Even output (butterfly add)
//     dst[k+N/2] = sub0[k] - t      // Odd output (butterfly subtract)
func combineRadix2[T Complex](
	dst []T,        // Output buffer (size N)
	sub0, sub1 []T, // Two N/2 sub-results
	twiddle []T,    // Twiddle factors W^k for k=0..N/2-1
) {
	half := len(sub0)
	for k := 0; k < half; k++ {
		t := twiddle[k] * sub1[k] // Twiddle multiplication
		dst[k] = sub0[k] + t      // Even output
		dst[k+half] = sub0[k] - t // Odd output
	}
}

// combineRadix4 combines four N/4-point FFTs into an N-point FFT.
// This is the radix-4 decimation-in-time combine step.
//
// Algorithm (DIT radix-4 butterfly):
//   For k = 0 to N/4-1:
//     t1 = W^k     * sub1[k]
//     t2 = W^(2k)  * sub2[k]
//     t3 = W^(3k)  * sub3[k]
//
//     dst[k + 0*N/4] = sub0[k] + t1 + t2 + t3        // Output bin 0
//     dst[k + 1*N/4] = sub0[k] - i*t1 - t2 + i*t3    // Output bin 1
//     dst[k + 2*N/4] = sub0[k] - t1 + t2 - t3        // Output bin 2
//     dst[k + 3*N/4] = sub0[k] + i*t1 - t2 - i*t3    // Output bin 3
//
// Note: W^(2k) and W^(3k) are precomputed and passed as twiddle2, twiddle3.
func combineRadix4[T Complex](
	dst []T,                       // Output buffer (size N)
	sub0, sub1, sub2, sub3 []T,    // Four N/4 sub-results
	twiddle1, twiddle2, twiddle3 []T, // Twiddle factors W^k, W^(2k), W^(3k)
) {
	quarter := len(sub0)

	for k := 0; k < quarter; k++ {
		t1 := twiddle1[k] * sub1[k]
		t2 := twiddle2[k] * sub2[k]
		t3 := twiddle3[k] * sub3[k]

		s0 := sub0[k]

		// Radix-4 butterfly (Gentleman-Sande decimation variant)
		// Multiplication by -i is equivalent to swapping real/imag and negating new real
		negIT1 := multiplyByNegI(t1)
		posIT3 := multiplyByI(t3)

		dst[k+0*quarter] = s0 + t1 + t2 + t3
		dst[k+1*quarter] = s0 + negIT1 - t2 + posIT3
		dst[k+2*quarter] = s0 - t1 + t2 - t3
		dst[k+3*quarter] = s0 - negIT1 - t2 - posIT3
	}
}

// combineRadix8 combines eight N/8-point FFTs into an N-point FFT.
// This is the radix-8 decimation-in-time combine step.
func combineRadix8[T Complex](
	dst []T,     // Output buffer (size N)
	subs [][]T,  // Eight N/8 sub-results (subs[0] to subs[7])
	twiddles [][]T, // Twiddle factors: twiddles[r][k] = W^(r*k) for r=0..7
) {
	eighth := len(subs[0])

	for k := 0; k < eighth; k++ {
		// Apply twiddle factors
		t := make([]T, 8)
		t[0] = subs[0][k] // W^0 = 1, no multiplication needed
		for r := 1; r < 8; r++ {
			t[r] = twiddles[r][k] * subs[r][k]
		}

		// Radix-8 butterfly (can be optimized further with radix-2 + radix-4 decomposition)
		// For now, use direct DFT formula for radix-8
		for bin := 0; bin < 8; bin++ {
			sum := T(0)
			for r := 0; r < 8; r++ {
				// W_8^(bin*r) rotation
				angle := -2.0 * 3.14159265358979323846 * float64(bin*r) / 8.0
				w := T(complex(cos64(angle), sin64(angle)))
				sum += w * t[r]
			}
			dst[k+bin*eighth] = sum
		}
	}
}

// combineGeneral combines an arbitrary number of sub-FFTs.
// This is a fallback for unusual radix values.
func combineGeneral[T Complex](
	dst []T,       // Output buffer (size N)
	subs [][]T,    // Radix sub-results (each of size N/radix)
	twiddles [][]T, // Twiddle factors: twiddles[r][k] = W^(r*k)
	radix int,
) {
	subSize := len(subs[0])

	for k := 0; k < subSize; k++ {
		// Apply twiddle factors
		t := make([]T, radix)
		t[0] = subs[0][k]
		for r := 1; r < radix; r++ {
			t[r] = twiddles[r][k] * subs[r][k]
		}

		// General DFT for this radix
		for bin := 0; bin < radix; bin++ {
			sum := T(0)
			for r := 0; r < radix; r++ {
				angle := -2.0 * 3.14159265358979323846 * float64(bin*r) / float64(radix)
				w := T(complex(cos64(angle), sin64(angle)))
				sum += w * t[r]
			}
			dst[k+bin*subSize] = sum
		}
	}
}

// multiplyByI multiplies a complex number by i (90° rotation).
// i * (a + bi) = -b + ai
func multiplyByI[T Complex](x T) T {
	// Multiply by i: rotate 90 degrees counterclockwise
	// This is equivalent to: x * complex(0, 1)
	var zero T
	switch any(zero).(type) {
	case complex64:
		return any(any(x).(complex64) * complex64(complex(0, 1))).(T)
	case complex128:
		return any(any(x).(complex128) * complex128(complex(0, 1))).(T)
	default:
		panic("unsupported complex type")
	}
}

// multiplyByNegI multiplies a complex number by -i (-90° rotation).
// -i * (a + bi) = b - ai
func multiplyByNegI[T Complex](x T) T {
	// Multiply by -i: rotate 90 degrees clockwise
	// This is equivalent to: x * complex(0, -1)
	var zero T
	switch any(zero).(type) {
	case complex64:
		return any(any(x).(complex64) * complex64(complex(0, -1))).(T)
	case complex128:
		return any(any(x).(complex128) * complex128(complex(0, -1))).(T)
	default:
		panic("unsupported complex type")
	}
}

// cos64 returns cosine of x (float64 input).
func cos64(x float64) float64 {
	return math.Cos(x)
}

// sin64 returns sine of x (float64 input).
func sin64(x float64) float64 {
	return math.Sin(x)
}
