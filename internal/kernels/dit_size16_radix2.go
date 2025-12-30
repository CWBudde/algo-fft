package kernels

// forwardDIT16Complex64 computes a 16-point forward FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3, w4, w5, w6, w7 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7]

	// Stage 1: 8 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1

	// Stage 2: 4 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w4 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w4 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w4 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w4 * a15
	b13, b15 := a13+t, a13-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w2 * b5
	c1, c5 := b1+t, b1-t
	t = w4 * b6
	c2, c6 := b2+t, b2-t
	t = w6 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w2 * b13
	c9, c13 := b9+t, b9-t
	t = w4 * b14
	c10, c14 := b10+t, b10-t
	t = w6 * b15
	c11, c15 := b11+t, b11-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[8] = c0+c8, c0-c8
	t = w1 * c9
	work[1], work[9] = c1+t, c1-t
	t = w2 * c10
	work[2], work[10] = c2+t, c2-t
	t = w3 * c11
	work[3], work[11] = c3+t, c3-t
	t = w4 * c12
	work[4], work[12] = c4+t, c4-t
	t = w5 * c13
	work[5], work[13] = c5+t, c5-t
	t = w6 * c14
	work[6], work[14] = c6+t, c6-t
	t = w7 * c15
	work[7], work[15] = c7+t, c7-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT16Complex64 computes a 16-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT16Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3, w4, w5, w6, w7 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))
	w4 = complex(real(w4), -imag(w4))
	w5 = complex(real(w5), -imag(w5))
	w6 = complex(real(w6), -imag(w6))
	w7 = complex(real(w7), -imag(w7))

	// Stage 1: 8 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1

	// Stage 2: 4 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w4 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w4 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w4 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w4 * a15
	b13, b15 := a13+t, a13-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w2 * b5
	c1, c5 := b1+t, b1-t
	t = w4 * b6
	c2, c6 := b2+t, b2-t
	t = w6 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w2 * b13
	c9, c13 := b9+t, b9-t
	t = w4 * b14
	c10, c14 := b10+t, b10-t
	t = w6 * b15
	c11, c15 := b11+t, b11-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[8] = c0+c8, c0-c8
	t = w1 * c9
	work[1], work[9] = c1+t, c1-t
	t = w2 * c10
	work[2], work[10] = c2+t, c2-t
	t = w3 * c11
	work[3], work[11] = c3+t, c3-t
	t = w4 * c12
	work[4], work[12] = c4+t, c4-t
	t = w5 * c13
	work[5], work[13] = c5+t, c5-t
	t = w6 * c14
	work[6], work[14] = c6+t, c6-t
	t = w7 * c15
	work[7], work[15] = c7+t, c7-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT16Complex128 computes a 16-point forward FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3, w4, w5, w6, w7 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7]

	// Stage 1: 8 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1

	// Stage 2: 4 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w4 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w4 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w4 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w4 * a15
	b13, b15 := a13+t, a13-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w2 * b5
	c1, c5 := b1+t, b1-t
	t = w4 * b6
	c2, c6 := b2+t, b2-t
	t = w6 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w2 * b13
	c9, c13 := b9+t, b9-t
	t = w4 * b14
	c10, c14 := b10+t, b10-t
	t = w6 * b15
	c11, c15 := b11+t, b11-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[8] = c0+c8, c0-c8
	t = w1 * c9
	work[1], work[9] = c1+t, c1-t
	t = w2 * c10
	work[2], work[10] = c2+t, c2-t
	t = w3 * c11
	work[3], work[11] = c3+t, c3-t
	t = w4 * c12
	work[4], work[12] = c4+t, c4-t
	t = w5 * c13
	work[5], work[13] = c5+t, c5-t
	t = w6 * c14
	work[6], work[14] = c6+t, c6-t
	t = w7 * c15
	work[7], work[15] = c7+t, c7-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT16Complex128 computes a 16-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
//
//nolint:funlen
func inverseDIT16Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3, w4, w5, w6, w7 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))
	w4 = complex(real(w4), -imag(w4))
	w5 = complex(real(w5), -imag(w5))
	w6 = complex(real(w6), -imag(w6))
	w7 = complex(real(w7), -imag(w7))

	// Stage 1: 8 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1

	// Stage 2: 4 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w4 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w4 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w4 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w4 * a15
	b13, b15 := a13+t, a13-t

	// Stage 3: 2 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w2 * b5
	c1, c5 := b1+t, b1-t
	t = w4 * b6
	c2, c6 := b2+t, b2-t
	t = w6 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w2 * b13
	c9, c13 := b9+t, b9-t
	t = w4 * b14
	c10, c14 := b10+t, b10-t
	t = w6 * b15
	c11, c15 := b11+t, b11-t

	// Stage 4: 1 radix-2 butterfly, stride=16 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[8] = c0+c8, c0-c8
	t = w1 * c9
	work[1], work[9] = c1+t, c1-t
	t = w2 * c10
	work[2], work[10] = c2+t, c2-t
	t = w3 * c11
	work[3], work[11] = c3+t, c3-t
	t = w4 * c12
	work[4], work[12] = c4+t, c4-t
	t = w5 * c13
	work[5], work[13] = c5+t, c5-t
	t = w6 * c14
	work[6], work[14] = c6+t, c6-t
	t = w7 * c15
	work[7], work[15] = c7+t, c7-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
