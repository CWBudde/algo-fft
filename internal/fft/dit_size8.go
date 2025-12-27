package fft

func forwardDIT8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	x0 := s[br[0]]
	x1 := s[br[1]]
	x2 := s[br[2]]
	x3 := s[br[3]]
	x4 := s[br[4]]
	x5 := s[br[5]]
	x6 := s[br[6]]
	x7 := s[br[7]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7

	// Stage 2 (size 4)
	w2 := twiddle[2]
	b0, b2 := a0+a2, a0-a2
	t1 := w2 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w2 * a7
	b5, b7 := a5+t5, a5-t5

	// Stage 3 (size 8)
	w1 := twiddle[1]
	w3 := twiddle[3]
	c0, c4 := b0+b4, b0-b4
	t1 = w1 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w2 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w3 * b7
	c3, c7 := b3+t3, b3-t3

	work = work[:n]
	work[0] = c0
	work[1] = c1
	work[2] = c2
	work[3] = c3
	work[4] = c4
	work[5] = c5
	work[6] = c6
	work[7] = c7

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	x0 := s[br[0]]
	x1 := s[br[1]]
	x2 := s[br[2]]
	x3 := s[br[3]]
	x4 := s[br[4]]
	x5 := s[br[5]]
	x6 := s[br[6]]
	x7 := s[br[7]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7

	// Stage 2 (size 4)
	w2 := twiddle[2]
	w2 = complex(real(w2), -imag(w2))
	b0, b2 := a0+a2, a0-a2
	t1 := w2 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w2 * a7
	b5, b7 := a5+t5, a5-t5

	// Stage 3 (size 8)
	w1 := twiddle[1]
	w1 = complex(real(w1), -imag(w1))
	w3 := twiddle[3]
	w3 = complex(real(w3), -imag(w3))
	c0, c4 := b0+b4, b0-b4
	t1 = w1 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w2 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w3 * b7
	c3, c7 := b3+t3, b3-t3

	work = work[:n]
	work[0] = c0
	work[1] = c1
	work[2] = c2
	work[3] = c3
	work[4] = c4
	work[5] = c5
	work[6] = c6
	work[7] = c7

	if !workIsDst {
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

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	x0 := s[br[0]]
	x1 := s[br[1]]
	x2 := s[br[2]]
	x3 := s[br[3]]
	x4 := s[br[4]]
	x5 := s[br[5]]
	x6 := s[br[6]]
	x7 := s[br[7]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7

	// Stage 2 (size 4)
	w2 := twiddle[2]
	b0, b2 := a0+a2, a0-a2
	t1 := w2 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w2 * a7
	b5, b7 := a5+t5, a5-t5

	// Stage 3 (size 8)
	w1 := twiddle[1]
	w3 := twiddle[3]
	c0, c4 := b0+b4, b0-b4
	t1 = w1 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w2 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w3 * b7
	c3, c7 := b3+t3, b3-t3

	work = work[:n]
	work[0] = c0
	work[1] = c1
	work[2] = c2
	work[3] = c3
	work[4] = c4
	work[5] = c5
	work[6] = c6
	work[7] = c7

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	x0 := s[br[0]]
	x1 := s[br[1]]
	x2 := s[br[2]]
	x3 := s[br[3]]
	x4 := s[br[4]]
	x5 := s[br[5]]
	x6 := s[br[6]]
	x7 := s[br[7]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7

	// Stage 2 (size 4)
	w2 := twiddle[2]
	w2 = complex(real(w2), -imag(w2))
	b0, b2 := a0+a2, a0-a2
	t1 := w2 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w2 * a7
	b5, b7 := a5+t5, a5-t5

	// Stage 3 (size 8)
	w1 := twiddle[1]
	w1 = complex(real(w1), -imag(w1))
	w3 := twiddle[3]
	w3 = complex(real(w3), -imag(w3))
	c0, c4 := b0+b4, b0-b4
	t1 = w1 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w2 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w3 * b7
	c3, c7 := b3+t3, b3-t3

	work = work[:n]
	work[0] = c0
	work[1] = c1
	work[2] = c2
	work[3] = c3
	work[4] = c4
	work[5] = c5
	work[6] = c6
	work[7] = c7

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
