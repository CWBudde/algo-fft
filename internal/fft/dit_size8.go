package fft

func forwardDIT8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	// Stage 1 (size 2) - with interleaved loads
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

	// Stage 2 (size 4)
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3 (size 8) - write directly to output
	// Use scratch if dst and src overlap
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

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

func inverseDIT8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Stage 1 (size 2) - with interleaved loads
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

	// Stage 2 (size 4)
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3 (size 8) - write directly to output
	// Use scratch if dst and src overlap
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

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	// Stage 1 (size 2) - with interleaved loads
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

	// Stage 2 (size 4)
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3 (size 8) - write directly to output
	// Use scratch if dst and src overlap
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

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

func inverseDIT8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	// Stage 1 (size 2) - with interleaved loads
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

	// Stage 2 (size 4)
	b0, b2 := a0+a2, a0-a2
	t := w2 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w2 * a7
	b5, b7 := a5+t, a5-t

	// Stage 3 (size 8) - write directly to output
	// Use scratch if dst and src overlap
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

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
