package kernels

import (
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// forwardDIT16Radix2Complex64 computes a 16-point forward FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Radix2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3, w4, w5, w6, w7 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7]

	work1 := scratch[:n]
	work2 := dst[:n]

	// Stage 1: 8 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[0]
	x1 := s[8]
	work1[0], work1[1] = x0+x1, x0-x1
	x0 = s[4]
	x1 = s[12]
	work1[2], work1[3] = x0+x1, x0-x1
	x0 = s[2]
	x1 = s[10]
	work1[4], work1[5] = x0+x1, x0-x1
	x0 = s[6]
	x1 = s[14]
	work1[6], work1[7] = x0+x1, x0-x1
	x0 = s[1]
	x1 = s[9]
	work1[8], work1[9] = x0+x1, x0-x1
	x0 = s[5]
	x1 = s[13]
	work1[10], work1[11] = x0+x1, x0-x1
	x0 = s[3]
	x1 = s[11]
	work1[12], work1[13] = x0+x1, x0-x1
	x0 = s[7]
	x1 = s[15]
	work1[14], work1[15] = x0+x1, x0-x1

	// Stage 2: 4 radix-2 butterflies, stride=4
	work2[0], work2[2] = work1[0]+work1[2], work1[0]-work1[2]
	t := mathpkg.MulComplex64(w4, work1[3])
	work2[1], work2[3] = work1[1]+t, work1[1]-t
	work2[4], work2[6] = work1[4]+work1[6], work1[4]-work1[6]
	t = mathpkg.MulComplex64(w4, work1[7])
	work2[5], work2[7] = work1[5]+t, work1[5]-t
	work2[8], work2[10] = work1[8]+work1[10], work1[8]-work1[10]
	t = mathpkg.MulComplex64(w4, work1[11])
	work2[9], work2[11] = work1[9]+t, work1[9]-t
	work2[12], work2[14] = work1[12]+work1[14], work1[12]-work1[14]
	t = mathpkg.MulComplex64(w4, work1[15])
	work2[13], work2[15] = work1[13]+t, work1[13]-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	work1[0], work1[4] = work2[0]+work2[4], work2[0]-work2[4]
	t = mathpkg.MulComplex64(w2, work2[5])
	work1[1], work1[5] = work2[1]+t, work2[1]-t
	t = mathpkg.MulComplex64(w4, work2[6])
	work1[2], work1[6] = work2[2]+t, work2[2]-t
	t = mathpkg.MulComplex64(w6, work2[7])
	work1[3], work1[7] = work2[3]+t, work2[3]-t
	work1[8], work1[12] = work2[8]+work2[12], work2[8]-work2[12]
	t = mathpkg.MulComplex64(w2, work2[13])
	work1[9], work1[13] = work2[9]+t, work2[9]-t
	t = mathpkg.MulComplex64(w4, work2[14])
	work1[10], work1[14] = work2[10]+t, work2[10]-t
	t = mathpkg.MulComplex64(w6, work2[15])
	work1[11], work1[15] = work2[11]+t, work2[11]-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	work2[0], work2[8] = work1[0]+work1[8], work1[0]-work1[8]
	t = mathpkg.MulComplex64(w1, work1[9])
	work2[1], work2[9] = work1[1]+t, work1[1]-t
	t = mathpkg.MulComplex64(w2, work1[10])
	work2[2], work2[10] = work1[2]+t, work1[2]-t
	t = mathpkg.MulComplex64(w3, work1[11])
	work2[3], work2[11] = work1[3]+t, work1[3]-t
	t = mathpkg.MulComplex64(w4, work1[12])
	work2[4], work2[12] = work1[4]+t, work1[4]-t
	t = mathpkg.MulComplex64(w5, work1[13])
	work2[5], work2[13] = work1[5]+t, work1[5]-t
	t = mathpkg.MulComplex64(w6, work1[14])
	work2[6], work2[14] = work1[6]+t, work1[6]-t
	t = mathpkg.MulComplex64(w7, work1[15])
	work2[7], work2[15] = work1[7]+t, work1[7]-t

	return true
}

// inverseDIT16Radix2Complex64 computes a 16-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT16Radix2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	s := src[:n]

	work1 := scratch[:n]
	work2 := dst[:n]

	// Stage 1: 8 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[0]
	x1 := s[8]
	work1[0], work1[1] = x0+x1, x0-x1
	x0 = s[4]
	x1 = s[12]
	work1[2], work1[3] = x0+x1, x0-x1
	x0 = s[2]
	x1 = s[10]
	work1[4], work1[5] = x0+x1, x0-x1

	x0 = s[6]
	x1 = s[14]
	work1[6], work1[7] = x0+x1, x0-x1

	x0 = s[1]
	x1 = s[9]
	work1[8], work1[9] = x0+x1, x0-x1

	x0 = s[5]
	x1 = s[13]
	work1[10], work1[11] = x0+x1, x0-x1

	x0 = s[3]
	x1 = s[11]
	work1[12], work1[13] = x0+x1, x0-x1

	x0 = s[7]
	x1 = s[15]
	work1[14], work1[15] = x0+x1, x0-x1

	// Stage 2: 4 radix-2 butterflies, stride=4
	work2[0], work2[2] = work1[0]+work1[2], work1[0]-work1[2]
	w4 := twiddle[4]
	t := mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work1[3])
	work2[1], work2[3] = work1[1]+t, work1[1]-t
	work2[4], work2[6] = work1[4]+work1[6], work1[4]-work1[6]
	t = mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work1[7])
	work2[5], work2[7] = work1[5]+t, work1[5]-t
	work2[8], work2[10] = work1[8]+work1[10], work1[8]-work1[10]
	t = mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work1[11])
	work2[9], work2[11] = work1[9]+t, work1[9]-t
	work2[12], work2[14] = work1[12]+work1[14], work1[12]-work1[14]
	t = mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work1[15])
	work2[13], work2[15] = work1[13]+t, work1[13]-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	w2 := twiddle[2]
	w6 := twiddle[6]
	work1[0], work1[4] = work2[0]+work2[4], work2[0]-work2[4]
	t = mathpkg.MulComplex64(complex(real(w2), -imag(w2)), work2[5])
	work1[1], work1[5] = work2[1]+t, work2[1]-t
	t = mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work2[6])
	work1[2], work1[6] = work2[2]+t, work2[2]-t
	t = mathpkg.MulComplex64(complex(real(w6), -imag(w6)), work2[7])
	work1[3], work1[7] = work2[3]+t, work2[3]-t
	work1[8], work1[12] = work2[8]+work2[12], work2[8]-work2[12]
	t = mathpkg.MulComplex64(complex(real(w2), -imag(w2)), work2[13])
	work1[9], work1[13] = work2[9]+t, work2[9]-t
	t = mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work2[14])
	work1[10], work1[14] = work2[10]+t, work2[10]-t
	t = mathpkg.MulComplex64(complex(real(w6), -imag(w6)), work2[15])
	work1[11], work1[15] = work2[11]+t, work2[11]-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array), with the 1/n
	// normalisation folded in. Each butterfly's two operands are scaled once
	// rather than each of its two results, and by a real factor rather than by
	// complex(1/n, 0): four real multiplies per pair instead of eight plus four
	// adds. 1/n is a power of two here, so this is exact.
	scale := float32(1.0 / float64(n))
	w1, w3, w5, w7 := twiddle[1], twiddle[3], twiddle[5], twiddle[7]

	a := complex(real(work1[0])*scale, imag(work1[0])*scale)
	b := complex(real(work1[8])*scale, imag(work1[8])*scale)
	work2[0], work2[8] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w1), -imag(w1)), work1[9])
	a = complex(real(work1[1])*scale, imag(work1[1])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[1], work2[9] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w2), -imag(w2)), work1[10])
	a = complex(real(work1[2])*scale, imag(work1[2])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[2], work2[10] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w3), -imag(w3)), work1[11])
	a = complex(real(work1[3])*scale, imag(work1[3])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[3], work2[11] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w4), -imag(w4)), work1[12])
	a = complex(real(work1[4])*scale, imag(work1[4])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[4], work2[12] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w5), -imag(w5)), work1[13])
	a = complex(real(work1[5])*scale, imag(work1[5])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[5], work2[13] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w6), -imag(w6)), work1[14])
	a = complex(real(work1[6])*scale, imag(work1[6])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[6], work2[14] = a+b, a-b

	t = mathpkg.MulComplex64(complex(real(w7), -imag(w7)), work1[15])
	a = complex(real(work1[7])*scale, imag(work1[7])*scale)
	b = complex(real(t)*scale, imag(t)*scale)
	work2[7], work2[15] = a+b, a-b

	return true
}
