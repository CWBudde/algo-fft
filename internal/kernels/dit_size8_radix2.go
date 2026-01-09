package kernels

// bitrev8Radix2 is the precomputed bit-reversal indices for size 8 radix-2.
var bitrev8Radix2 = [8]int{0, 4, 2, 6, 1, 5, 3, 7}

// forwardDIT8Radix2Complex64 computes an 8-point forward FFT using the
// Decimation-in-Time (DIT) algorithm with radix-2 stages for complex64 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev8Radix2[:]
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
//
//nolint:funlen
func inverseDIT8Radix2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev8Radix2[:]
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
func forwardDIT8Radix2Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev8Radix2[:]
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
//
//nolint:funlen
func inverseDIT8Radix2Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev8Radix2[:]
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Stage 1: 4 radix-2 butterflies, stride=2, no twiddles (W^1 = 1)
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
