package transform

import (
	"math"

	imath "github.com/cwbudde/algo-fft/internal/math"
)

// combine.go implements the "combine" step of Cooley-Tukey decomposition.
// After computing sub-FFTs, these functions merge them using twiddle factors.

// combineRadix2 combines two N/2-point FFTs into an N-point FFT.
// This is the classic Cooley-Tukey radix-2 decimation-in-time combine step.
//
// Algorithm:
//
//	For k = 0 to N/2-1:
//	  t = twiddle[k] * sub1[k]     // Twiddle multiplication
//	  dst[k]     = sub0[k] + t      // Even output (butterfly add)
//	  dst[k+N/2] = sub0[k] - t      // Odd output (butterfly subtract)
func combineRadix2[T Complex](
	dst []T, // Output buffer (size N)
	sub0, sub1 []T, // Two N/2 sub-results
	twiddle []T, // Twiddle factors W^k for k=0..N/2-1
) {
	half := len(sub0)
	for k := range half {
		t := twiddle[k] * sub1[k] // Twiddle multiplication
		dst[k] = sub0[k] + t      // Even output
		dst[k+half] = sub0[k] - t // Odd output
	}
}

// combineRadix4 combines four N/4-point FFTs into an N-point FFT.
// This is the radix-4 decimation-in-time combine step.
//
// Algorithm (DIT radix-4 butterfly):
//
//	For k = 0 to N/4-1:
//	  t1 = W^k     * sub1[k]
//	  t2 = W^(2k)  * sub2[k]
//	  t3 = W^(3k)  * sub3[k]
//
//	  dst[k + 0*N/4] = sub0[k] + t1 + t2 + t3        // Output bin 0
//	  dst[k + 1*N/4] = sub0[k] - i*t1 - t2 + i*t3    // Output bin 1
//	  dst[k + 2*N/4] = sub0[k] - t1 + t2 - t3        // Output bin 2
//	  dst[k + 3*N/4] = sub0[k] + i*t1 - t2 - i*t3    // Output bin 3
//
// Note: W^(2k) and W^(3k) are precomputed and passed as twiddle2, twiddle3.
func combineRadix4[T Complex](
	dst []T, // Output buffer (size N)
	sub0, sub1, sub2, sub3 []T, // Four N/4 sub-results
	twiddle1, twiddle2, twiddle3 []T, // Twiddle factors W^k, W^(2k), W^(3k)
) {
	quarter := len(sub0)

	for k := range quarter {
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
//
// subs and twiddles are flat blocks in [r][k] order: element (r, k) lives at
// r*subSize+k. That is already how the recursive scratch and twiddle tables
// are laid out, so no per-call slice-of-slice views are needed.
func combineRadix8[T Complex](
	dst []T, // Output buffer (size N)
	subs []T, // Eight N/8 sub-results, flat [r][k]
	twiddles []T, // Twiddle factors W^(r*k), flat [r][k]
	subSize int, // N/8
) {
	// W_8^j for j = 0..7. These depend only on the radix, so they are built
	// once per call; computing them inside the element loop cost 64 sin/cos
	// pairs per output element and dominated the whole transform.
	var roots [8]T

	for j := range roots {
		angle := -imath.TwoPi * float64(j) / 8.0
		roots[j] = T(complex(cos64(angle), sin64(angle)))
	}

	var t [8]T

	for k := range subSize {
		t[0] = subs[k] // W^0 = 1, no multiplication needed
		for r := 1; r < 8; r++ {
			t[r] = twiddles[r*subSize+k] * subs[r*subSize+k]
		}

		// Radix-8 butterfly, evaluated as a direct 8-point DFT over t.
		for bin := range 8 {
			sum := t[0]
			for r := 1; r < 8; r++ {
				sum += roots[(bin*r)&7] * t[r]
			}

			dst[k+bin*subSize] = sum
		}
	}
}

// combineGeneral combines an arbitrary number of sub-FFTs.
//
// This is a fallback for radices without a dedicated butterfly.
// PlanDecomposition never emits one (see combineRadices), so it serves only
// hand-built strategies; it stays allocation-free and evaluates each rotation
// once per call rather than once per output element.
func combineGeneral[T Complex](
	dst []T, // Output buffer (size N)
	subs []T, // Radix sub-results, flat [r][k]
	twiddles []T, // Twiddle factors W^(r*k), flat [r][k]
	subSize int, // N/radix
	radix int,
) {
	// Bin-outer ordering: the rotation W_radix^(bin*r) is loop-invariant in k,
	// so it is computed radix^2 times per call instead of per element. The
	// r == 0 term has W^0 = 1 and seeds the accumulator.
	for bin := range radix {
		out := dst[bin*subSize : (bin+1)*subSize]
		copy(out, subs[:subSize])

		for r := 1; r < radix; r++ {
			angle := -imath.TwoPi * float64((bin*r)%radix) / float64(radix)
			w := T(complex(cos64(angle), sin64(angle)))

			tw := twiddles[r*subSize : (r+1)*subSize]
			sub := subs[r*subSize : (r+1)*subSize]

			for k := range subSize {
				out[k] += w * (tw[k] * sub[k])
			}
		}
	}
}

// multiplyByI multiplies a complex number by i (90° rotation).
// i * (a + bi) = -b + ai.
func multiplyByI[T Complex](x T) T {
	// Multiply by i: rotate 90 degrees counterclockwise
	// This is equivalent to: x * complex(0, 1)
	switch xv := any(x).(type) {
	case complex64:
		res := xv * complex(0, 1)
		rv, _ := any(res).(T)

		return rv
	case complex128:
		res := xv * complex(0, 1)
		rv, _ := any(res).(T)

		return rv
	default:
		panic("unsupported complex type")
	}
}

// multiplyByNegI multiplies a complex number by -i (-90° rotation).
// -i * (a + bi) = b - ai.
func multiplyByNegI[T Complex](x T) T {
	// Multiply by -i: rotate 90 degrees clockwise
	// This is equivalent to: x * complex(0, -1)
	switch xv := any(x).(type) {
	case complex64:
		res := xv * complex(0, -1)
		rv, _ := any(res).(T)

		return rv
	case complex128:
		res := xv * complex(0, -1)
		rv, _ := any(res).(T)

		return rv
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
