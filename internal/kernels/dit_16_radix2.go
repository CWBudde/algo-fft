package kernels

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
	t := w4 * work1[3]
	work2[1], work2[3] = work1[1]+t, work1[1]-t
	work2[4], work2[6] = work1[4]+work1[6], work1[4]-work1[6]
	t = w4 * work1[7]
	work2[5], work2[7] = work1[5]+t, work1[5]-t
	work2[8], work2[10] = work1[8]+work1[10], work1[8]-work1[10]
	t = w4 * work1[11]
	work2[9], work2[11] = work1[9]+t, work1[9]-t
	work2[12], work2[14] = work1[12]+work1[14], work1[12]-work1[14]
	t = w4 * work1[15]
	work2[13], work2[15] = work1[13]+t, work1[13]-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	work1[0], work1[4] = work2[0]+work2[4], work2[0]-work2[4]
	t = w2 * work2[5]
	work1[1], work1[5] = work2[1]+t, work2[1]-t
	t = w4 * work2[6]
	work1[2], work1[6] = work2[2]+t, work2[2]-t
	t = w6 * work2[7]
	work1[3], work1[7] = work2[3]+t, work2[3]-t
	work1[8], work1[12] = work2[8]+work2[12], work2[8]-work2[12]
	t = w2 * work2[13]
	work1[9], work1[13] = work2[9]+t, work2[9]-t
	t = w4 * work2[14]
	work1[10], work1[14] = work2[10]+t, work2[10]-t
	t = w6 * work2[15]
	work1[11], work1[15] = work2[11]+t, work2[11]-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	work2[0], work2[8] = work1[0]+work1[8], work1[0]-work1[8]
	t = w1 * work1[9]
	work2[1], work2[9] = work1[1]+t, work1[1]-t
	t = w2 * work1[10]
	work2[2], work2[10] = work1[2]+t, work1[2]-t
	t = w3 * work1[11]
	work2[3], work2[11] = work1[3]+t, work1[3]-t
	t = w4 * work1[12]
	work2[4], work2[12] = work1[4]+t, work1[4]-t
	t = w5 * work1[13]
	work2[5], work2[13] = work1[5]+t, work1[5]-t
	t = w6 * work1[14]
	work2[6], work2[14] = work1[6]+t, work1[6]-t
	t = w7 * work1[15]
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
	t := complex(real(w4), -imag(w4)) * work1[3]
	work2[1], work2[3] = work1[1]+t, work1[1]-t
	work2[4], work2[6] = work1[4]+work1[6], work1[4]-work1[6]
	t = complex(real(w4), -imag(w4)) * work1[7]
	work2[5], work2[7] = work1[5]+t, work1[5]-t
	work2[8], work2[10] = work1[8]+work1[10], work1[8]-work1[10]
	t = complex(real(w4), -imag(w4)) * work1[11]
	work2[9], work2[11] = work1[9]+t, work1[9]-t
	work2[12], work2[14] = work1[12]+work1[14], work1[12]-work1[14]
	t = complex(real(w4), -imag(w4)) * work1[15]
	work2[13], work2[15] = work1[13]+t, work1[13]-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	w2 := twiddle[2]
	w6 := twiddle[6]
	work1[0], work1[4] = work2[0]+work2[4], work2[0]-work2[4]
	t = complex(real(w2), -imag(w2)) * work2[5]
	work1[1], work1[5] = work2[1]+t, work2[1]-t
	t = complex(real(w4), -imag(w4)) * work2[6]
	work1[2], work1[6] = work2[2]+t, work2[2]-t
	t = complex(real(w6), -imag(w6)) * work2[7]
	work1[3], work1[7] = work2[3]+t, work2[3]-t
	work1[8], work1[12] = work2[8]+work2[12], work2[8]-work2[12]
	t = complex(real(w2), -imag(w2)) * work2[13]
	work1[9], work1[13] = work2[9]+t, work2[9]-t
	t = complex(real(w4), -imag(w4)) * work2[14]
	work1[10], work1[14] = work2[10]+t, work2[10]-t
	t = complex(real(w6), -imag(w6)) * work2[15]
	work1[11], work1[15] = work2[11]+t, work2[11]-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	scale := complex(float32(1.0/float64(n)), 0)
	w1, w3, w5, w7 := twiddle[1], twiddle[3], twiddle[5], twiddle[7]
	work2[0] = (work1[0] + work1[8]) * scale
	work2[8] = (work1[0] - work1[8]) * scale
	t = complex(real(w1), -imag(w1)) * work1[9]
	work2[1] = (work1[1] + t) * scale
	work2[9] = (work1[1] - t) * scale
	t = complex(real(w2), -imag(w2)) * work1[10]
	work2[2] = (work1[2] + t) * scale
	work2[10] = (work1[2] - t) * scale
	t = complex(real(w3), -imag(w3)) * work1[11]
	work2[3] = (work1[3] + t) * scale
	work2[11] = (work1[3] - t) * scale
	t = complex(real(w4), -imag(w4)) * work1[12]
	work2[4] = (work1[4] + t) * scale
	work2[12] = (work1[4] - t) * scale
	t = complex(real(w5), -imag(w5)) * work1[13]
	work2[5] = (work1[5] + t) * scale
	work2[13] = (work1[5] - t) * scale
	t = complex(real(w6), -imag(w6)) * work1[14]
	work2[6] = (work1[6] + t) * scale
	work2[14] = (work1[6] - t) * scale
	t = complex(real(w7), -imag(w7)) * work1[15]
	work2[7] = (work1[7] + t) * scale
	work2[15] = (work1[7] - t) * scale

	return true
}

// forwardDIT16Radix2Complex128 computes a 16-point forward FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Radix2Complex128(dst, src, twiddle, scratch []complex128) bool {
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
	t := w4 * work1[3]
	work2[1], work2[3] = work1[1]+t, work1[1]-t
	work2[4], work2[6] = work1[4]+work1[6], work1[4]-work1[6]
	t = w4 * work1[7]
	work2[5], work2[7] = work1[5]+t, work1[5]-t
	work2[8], work2[10] = work1[8]+work1[10], work1[8]-work1[10]
	t = w4 * work1[11]
	work2[9], work2[11] = work1[9]+t, work1[9]-t
	work2[12], work2[14] = work1[12]+work1[14], work1[12]-work1[14]
	t = w4 * work1[15]
	work2[13], work2[15] = work1[13]+t, work1[13]-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	work1[0], work1[4] = work2[0]+work2[4], work2[0]-work2[4]
	t = w2 * work2[5]
	work1[1], work1[5] = work2[1]+t, work2[1]-t
	t = w4 * work2[6]
	work1[2], work1[6] = work2[2]+t, work2[2]-t
	t = w6 * work2[7]
	work1[3], work1[7] = work2[3]+t, work2[3]-t
	work1[8], work1[12] = work2[8]+work2[12], work2[8]-work2[12]
	t = w2 * work2[13]
	work1[9], work1[13] = work2[9]+t, work2[9]-t
	t = w4 * work2[14]
	work1[10], work1[14] = work2[10]+t, work2[10]-t
	t = w6 * work2[15]
	work1[11], work1[15] = work2[11]+t, work2[11]-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	work2[0], work2[8] = work1[0]+work1[8], work1[0]-work1[8]
	t = w1 * work1[9]
	work2[1], work2[9] = work1[1]+t, work1[1]-t
	t = w2 * work1[10]
	work2[2], work2[10] = work1[2]+t, work1[2]-t
	t = w3 * work1[11]
	work2[3], work2[11] = work1[3]+t, work1[3]-t
	t = w4 * work1[12]
	work2[4], work2[12] = work1[4]+t, work1[4]-t
	t = w5 * work1[13]
	work2[5], work2[13] = work1[5]+t, work1[5]-t
	t = w6 * work1[14]
	work2[6], work2[14] = work1[6]+t, work1[6]-t
	t = w7 * work1[15]
	work2[7], work2[15] = work1[7]+t, work1[7]-t

	return true
}

// inverseDIT16Radix2Complex128 computes a 16-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
//
//nolint:funlen
func inverseDIT16Radix2Complex128(dst, src, twiddle, scratch []complex128) bool {
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
	t := complex(real(w4), -imag(w4)) * work1[3]
	work2[1], work2[3] = work1[1]+t, work1[1]-t
	work2[4], work2[6] = work1[4]+work1[6], work1[4]-work1[6]
	t = complex(real(w4), -imag(w4)) * work1[7]
	work2[5], work2[7] = work1[5]+t, work1[5]-t
	work2[8], work2[10] = work1[8]+work1[10], work1[8]-work1[10]
	t = complex(real(w4), -imag(w4)) * work1[11]
	work2[9], work2[11] = work1[9]+t, work1[9]-t
	work2[12], work2[14] = work1[12]+work1[14], work1[12]-work1[14]
	t = complex(real(w4), -imag(w4)) * work1[15]
	work2[13], work2[15] = work1[13]+t, work1[13]-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	w2 := twiddle[2]
	w6 := twiddle[6]
	work1[0], work1[4] = work2[0]+work2[4], work2[0]-work2[4]
	t = complex(real(w2), -imag(w2)) * work2[5]
	work1[1], work1[5] = work2[1]+t, work2[1]-t
	t = complex(real(w4), -imag(w4)) * work2[6]
	work1[2], work1[6] = work2[2]+t, work2[2]-t
	t = complex(real(w6), -imag(w6)) * work2[7]
	work1[3], work1[7] = work2[3]+t, work2[3]-t
	work1[8], work1[12] = work2[8]+work2[12], work2[8]-work2[12]
	t = complex(real(w2), -imag(w2)) * work2[13]
	work1[9], work1[13] = work2[9]+t, work2[9]-t
	t = complex(real(w4), -imag(w4)) * work2[14]
	work1[10], work1[14] = work2[10]+t, work2[10]-t
	t = complex(real(w6), -imag(w6)) * work2[15]
	work1[11], work1[15] = work2[11]+t, work2[11]-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	scale := complex(1.0/float64(n), 0)
	w1, w3, w5, w7 := twiddle[1], twiddle[3], twiddle[5], twiddle[7]

	work2[0] = (work1[0] + work1[8]) * scale
	work2[8] = (work1[0] - work1[8]) * scale
	t = complex(real(w1), -imag(w1)) * work1[9]
	work2[1] = (work1[1] + t) * scale
	work2[9] = (work1[1] - t) * scale
	t = complex(real(w2), -imag(w2)) * work1[10]
	work2[2] = (work1[2] + t) * scale
	work2[10] = (work1[2] - t) * scale
	t = complex(real(w3), -imag(w3)) * work1[11]
	work2[3] = (work1[3] + t) * scale
	work2[11] = (work1[3] - t) * scale
	t = complex(real(w4), -imag(w4)) * work1[12]
	work2[4] = (work1[4] + t) * scale
	work2[12] = (work1[4] - t) * scale
	t = complex(real(w5), -imag(w5)) * work1[13]
	work2[5] = (work1[5] + t) * scale
	work2[13] = (work1[5] - t) * scale
	t = complex(real(w6), -imag(w6)) * work1[14]
	work2[6] = (work1[6] + t) * scale
	work2[14] = (work1[6] - t) * scale
	t = complex(real(w7), -imag(w7)) * work1[15]
	work2[7] = (work1[7] + t) * scale
	work2[15] = (work1[7] - t) * scale

	return true
}
