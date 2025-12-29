package fft

// forwardDIT8Radix2Complex64 computes an 8-point forward FFT using the
// Decimation-in-Time (DIT) algorithm with radix-2 stages for complex64 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	// Stage 1: 4 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
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

	// Stage 2: 2 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3: 1 radix-2 butterfly, stride=8 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]
	work[0], work[4] = b0+b4, b0-b4
	t = w1 * b5
	work[1], work[5] = b1+t, b1-t
	t = w2 * b6
	work[2], work[6] = b2+t, b2-t
	t = w3 * b7
	work[3], work[7] = b3+t, b3-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8Radix2Complex64 computes an 8-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm with radix-2 stages for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT8Radix2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Stage 1: 4 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
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

	// Stage 2: 2 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3: 1 radix-2 butterfly, stride=8 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]
	work[0], work[4] = b0+b4, b0-b4
	t = w1 * b5
	work[1], work[5] = b1+t, b1-t
	t = w2 * b6
	work[2], work[6] = b2+t, b2-t
	t = w3 * b7
	work[3], work[7] = b3+t, b3-t

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

// forwardDIT8Radix2Complex128 computes an 8-point forward FFT using the
// Decimation-in-Time (DIT) algorithm with radix-2 stages for complex128 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	// Stage 1: 4 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
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

	// Stage 2: 2 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3: 1 radix-2 butterfly, stride=8 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]
	work[0], work[4] = b0+b4, b0-b4
	t = w1 * b5
	work[1], work[5] = b1+t, b1-t
	t = w2 * b6
	work[2], work[6] = b2+t, b2-t
	t = w3 * b7
	work[3], work[7] = b3+t, b3-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8Radix2Complex128 computes an 8-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm with radix-2 stages for complex128 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT8Radix2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Stage 1: 4 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
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

	// Stage 2: 2 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3: 1 radix-2 butterfly, stride=8 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]
	work[0], work[4] = b0+b4, b0-b4
	t = w1 * b5
	work[1], work[5] = b1+t, b1-t
	t = w2 * b6
	work[2], work[6] = b2+t, b2-t
	t = w3 * b7
	work[3], work[7] = b3+t, b3-t

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

// Wrapper functions for compatibility with existing code

func forwardDIT8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT8Radix2Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseDIT8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT8Radix2Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardDIT8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseDIT8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

// forwardDIT8Radix4Complex64 computes an 8-point forward FFT using a mixed-radix
// approach: one radix-4 stage followed by one radix-2 stage for complex64 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Pre-load twiddle factors
	// For radix-4: w2 = e^(-2πi*2/8) = e^(-πi/2), w4 = e^(-2πi*4/8) = e^(-πi)
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	// Stage 1: 2 radix-4 butterflies
	// Radix-4 butterfly indices for size 8: stride=2
	// Butterfly 1: indices [0, 2, 4, 6]
	// Butterfly 2: indices [1, 3, 5, 7]

	// Load bit-reversed inputs
	x0 := s[br[0]]
	x2 := s[br[2]]
	x4 := s[br[4]]
	x6 := s[br[6]]

	x1 := s[br[1]]
	x3 := s[br[3]]
	x5 := s[br[5]]
	x7 := s[br[7]]

	// Radix-4 butterfly 1: [x0, x2, x4, x6]
	// w^0, w^2, w^4, w^6 where w = e^(-2πi/8)
	// w^0 = 1, w^2 = -i, w^4 = -1, w^6 = i
	t0 := x0 + x4 // x0 + x4*w^0
	t1 := x0 - x4 // x0 - x4*w^0
	t2 := x2 + x6 // x2 + x6*w^0
	t3 := x2 - x6 // x2 - x6*w^0

	// Apply w^2 = -i to middle terms: multiply by -i means (r,i) -> (i,-r)
	t3i := complex(-imag(t3), real(t3)) // t3 * (-i)

	a0 := t0 + t2
	a1 := t1 + t3i
	a2 := t0 - t2
	a3 := t1 - t3i

	// Radix-4 butterfly 2: [x1, x3, x5, x7]
	t0 = x1 + x5
	t1 = x1 - x5
	t2 = x3 + x7
	t3 = x3 - x7
	t3i = complex(-imag(t3), real(t3)) // t3 * (-i)

	a4 := t0 + t2
	a5 := t1 + t3i
	a6 := t0 - t2
	a7 := t1 - t3i

	// Stage 2: 4 radix-2 butterflies with twiddle factors
	// Combine outputs from the two radix-4 butterflies
	// (a0, a4) with w^0, (a1, a5) with w^1, (a2, a6) with w^2, (a3, a7) with w^3

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = a0 + a4 // w^0 = 1
	work[4] = a0 - a4

	t := w1 * a5
	work[1] = a1 + t
	work[5] = a1 - t

	t = w2 * a6
	work[2] = a2 + t
	work[6] = a2 - t

	t = w3 * a7
	work[3] = a3 + t
	work[7] = a3 - t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8Radix4Complex64 computes an 8-point inverse FFT using a mixed-radix
// approach: one radix-4 stage followed by one radix-2 stage for complex64 data.
// Uses conjugated twiddle factors and applies 1/N scaling at the end.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT8Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Load bit-reversed inputs
	x0 := s[br[0]]
	x2 := s[br[2]]
	x4 := s[br[4]]
	x6 := s[br[6]]

	x1 := s[br[1]]
	x3 := s[br[3]]
	x5 := s[br[5]]
	x7 := s[br[7]]

	// Radix-4 butterfly 1 with conjugated twiddles
	// For inverse: w^2 = -i becomes conj(-i) = i
	t0 := x0 + x4
	t1 := x0 - x4
	t2 := x2 + x6
	t3 := x2 - x6

	// Apply conj(w^2) = i to middle terms: multiply by i means (r,i) -> (-i,r)
	t3i := complex(imag(t3), -real(t3)) // t3 * i

	a0 := t0 + t2
	a1 := t1 + t3i
	a2 := t0 - t2
	a3 := t1 - t3i

	// Radix-4 butterfly 2
	t0 = x1 + x5
	t1 = x1 - x5
	t2 = x3 + x7
	t3 = x3 - x7
	t3i = complex(imag(t3), -real(t3)) // t3 * i

	a4 := t0 + t2
	a5 := t1 + t3i
	a6 := t0 - t2
	a7 := t1 - t3i

	// Stage 2: radix-2 butterflies with conjugated twiddle factors
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = a0 + a4
	work[4] = a0 - a4

	t := w1 * a5
	work[1] = a1 + t
	work[5] = a1 - t

	t = w2 * a6
	work[2] = a2 + t
	work[6] = a2 - t

	t = w3 * a7
	work[3] = a3 + t
	work[7] = a3 - t

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

// forwardDIT8Radix4Complex128 computes an 8-point forward FFT using a mixed-radix
// approach: one radix-4 stage followed by one radix-2 stage for complex128 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	// Load bit-reversed inputs
	x0 := s[br[0]]
	x2 := s[br[2]]
	x4 := s[br[4]]
	x6 := s[br[6]]

	x1 := s[br[1]]
	x3 := s[br[3]]
	x5 := s[br[5]]
	x7 := s[br[7]]

	// Radix-4 butterfly 1: [x0, x2, x4, x6]
	t0 := x0 + x4
	t1 := x0 - x4
	t2 := x2 + x6
	t3 := x2 - x6

	// Apply w^2 = -i: multiply by -i means (r,i) -> (i,-r)
	t3i := complex(-imag(t3), real(t3))

	a0 := t0 + t2
	a1 := t1 + t3i
	a2 := t0 - t2
	a3 := t1 - t3i

	// Radix-4 butterfly 2: [x1, x3, x5, x7]
	t0 = x1 + x5
	t1 = x1 - x5
	t2 = x3 + x7
	t3 = x3 - x7
	t3i = complex(-imag(t3), real(t3))

	a4 := t0 + t2
	a5 := t1 + t3i
	a6 := t0 - t2
	a7 := t1 - t3i

	// Stage 2: radix-2 butterflies with twiddle factors
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = a0 + a4
	work[4] = a0 - a4

	t := w1 * a5
	work[1] = a1 + t
	work[5] = a1 - t

	t = w2 * a6
	work[2] = a2 + t
	work[6] = a2 - t

	t = w3 * a7
	work[3] = a3 + t
	work[7] = a3 - t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8Radix4Complex128 computes an 8-point inverse FFT using a mixed-radix
// approach: one radix-4 stage followed by one radix-2 stage for complex128 data.
// Uses conjugated twiddle factors and applies 1/N scaling at the end.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT8Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev[:n]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Load bit-reversed inputs
	x0 := s[br[0]]
	x2 := s[br[2]]
	x4 := s[br[4]]
	x6 := s[br[6]]

	x1 := s[br[1]]
	x3 := s[br[3]]
	x5 := s[br[5]]
	x7 := s[br[7]]

	// Radix-4 butterfly 1 with conjugated twiddles
	t0 := x0 + x4
	t1 := x0 - x4
	t2 := x2 + x6
	t3 := x2 - x6

	// Apply conj(w^2) = i: multiply by i means (r,i) -> (-i,r)
	t3i := complex(imag(t3), -real(t3))

	a0 := t0 + t2
	a1 := t1 + t3i
	a2 := t0 - t2
	a3 := t1 - t3i

	// Radix-4 butterfly 2
	t0 = x1 + x5
	t1 = x1 - x5
	t2 = x3 + x7
	t3 = x3 - x7
	t3i = complex(imag(t3), -real(t3))

	a4 := t0 + t2
	a5 := t1 + t3i
	a6 := t0 - t2
	a7 := t1 - t3i

	// Stage 2: radix-2 butterflies with conjugated twiddle factors
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = a0 + a4
	work[4] = a0 - a4

	t := w1 * a5
	work[1] = a1 + t
	work[5] = a1 - t

	t = w2 * a6
	work[2] = a2 + t
	work[6] = a2 - t

	t = w3 * a7
	work[3] = a3 + t
	work[7] = a3 - t

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
