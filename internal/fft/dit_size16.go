package fft

func forwardDIT16Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16

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
	x8 := s[br[8]]
	x9 := s[br[9]]
	x10 := s[br[10]]
	x11 := s[br[11]]
	x12 := s[br[12]]
	x13 := s[br[13]]
	x14 := s[br[14]]
	x15 := s[br[15]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15

	// Stage 2 (size 4)
	w4 := twiddle[4]
	b0, b2 := a0+a2, a0-a2
	t1 := w4 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w4 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w4 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w4 * a15
	b13, b15 := a13+t13, a13-t13

	// Stage 3 (size 8)
	w2 := twiddle[2]
	w6 := twiddle[6]
	c0, c4 := b0+b4, b0-b4
	t1 = w2 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w4 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w6 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w2 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w4 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w6 * b15
	c11, c15 := b11+t11, b11-t11

	// Stage 4 (size 16)
	w1 := twiddle[1]
	w3 := twiddle[3]
	w5 := twiddle[5]
	w7 := twiddle[7]
	d0, d8 := c0+c8, c0-c8
	t1 = w1 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w2 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w3 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w4 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w5 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w6 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w7 * c15
	d7, d15 := c7+t7, c7-t7

	work = work[:n]
	work[0] = d0
	work[1] = d1
	work[2] = d2
	work[3] = d3
	work[4] = d4
	work[5] = d5
	work[6] = d6
	work[7] = d7
	work[8] = d8
	work[9] = d9
	work[10] = d10
	work[11] = d11
	work[12] = d12
	work[13] = d13
	work[14] = d14
	work[15] = d15

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT16Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16

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
	x8 := s[br[8]]
	x9 := s[br[9]]
	x10 := s[br[10]]
	x11 := s[br[11]]
	x12 := s[br[12]]
	x13 := s[br[13]]
	x14 := s[br[14]]
	x15 := s[br[15]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15

	// Stage 2 (size 4)
	w4 := twiddle[4]
	w4 = complex(real(w4), -imag(w4))
	b0, b2 := a0+a2, a0-a2
	t1 := w4 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w4 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w4 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w4 * a15
	b13, b15 := a13+t13, a13-t13

	// Stage 3 (size 8)
	w2 := twiddle[2]
	w2 = complex(real(w2), -imag(w2))
	w6 := twiddle[6]
	w6 = complex(real(w6), -imag(w6))
	c0, c4 := b0+b4, b0-b4
	t1 = w2 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w4 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w6 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w2 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w4 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w6 * b15
	c11, c15 := b11+t11, b11-t11

	// Stage 4 (size 16)
	w1 := twiddle[1]
	w1 = complex(real(w1), -imag(w1))
	w3 := twiddle[3]
	w3 = complex(real(w3), -imag(w3))
	w5 := twiddle[5]
	w5 = complex(real(w5), -imag(w5))
	w7 := twiddle[7]
	w7 = complex(real(w7), -imag(w7))
	d0, d8 := c0+c8, c0-c8
	t1 = w1 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w2 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w3 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w4 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w5 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w6 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w7 * c15
	d7, d15 := c7+t7, c7-t7

	work = work[:n]
	work[0] = d0
	work[1] = d1
	work[2] = d2
	work[3] = d3
	work[4] = d4
	work[5] = d5
	work[6] = d6
	work[7] = d7
	work[8] = d8
	work[9] = d9
	work[10] = d10
	work[11] = d11
	work[12] = d12
	work[13] = d13
	work[14] = d14
	work[15] = d15

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT16Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16

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
	x8 := s[br[8]]
	x9 := s[br[9]]
	x10 := s[br[10]]
	x11 := s[br[11]]
	x12 := s[br[12]]
	x13 := s[br[13]]
	x14 := s[br[14]]
	x15 := s[br[15]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15

	// Stage 2 (size 4)
	w4 := twiddle[4]
	b0, b2 := a0+a2, a0-a2
	t1 := w4 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w4 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w4 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w4 * a15
	b13, b15 := a13+t13, a13-t13

	// Stage 3 (size 8)
	w2 := twiddle[2]
	w6 := twiddle[6]
	c0, c4 := b0+b4, b0-b4
	t1 = w2 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w4 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w6 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w2 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w4 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w6 * b15
	c11, c15 := b11+t11, b11-t11

	// Stage 4 (size 16)
	w1 := twiddle[1]
	w3 := twiddle[3]
	w5 := twiddle[5]
	w7 := twiddle[7]
	d0, d8 := c0+c8, c0-c8
	t1 = w1 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w2 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w3 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w4 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w5 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w6 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w7 * c15
	d7, d15 := c7+t7, c7-t7

	work = work[:n]
	work[0] = d0
	work[1] = d1
	work[2] = d2
	work[3] = d3
	work[4] = d4
	work[5] = d5
	work[6] = d6
	work[7] = d7
	work[8] = d8
	work[9] = d9
	work[10] = d10
	work[11] = d11
	work[12] = d12
	work[13] = d13
	work[14] = d14
	work[15] = d15

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT16Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16

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
	x8 := s[br[8]]
	x9 := s[br[9]]
	x10 := s[br[10]]
	x11 := s[br[11]]
	x12 := s[br[12]]
	x13 := s[br[13]]
	x14 := s[br[14]]
	x15 := s[br[15]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15

	// Stage 2 (size 4)
	w4 := twiddle[4]
	w4 = complex(real(w4), -imag(w4))
	b0, b2 := a0+a2, a0-a2
	t1 := w4 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w4 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w4 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w4 * a15
	b13, b15 := a13+t13, a13-t13

	// Stage 3 (size 8)
	w2 := twiddle[2]
	w2 = complex(real(w2), -imag(w2))
	w6 := twiddle[6]
	w6 = complex(real(w6), -imag(w6))
	c0, c4 := b0+b4, b0-b4
	t1 = w2 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w4 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w6 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w2 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w4 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w6 * b15
	c11, c15 := b11+t11, b11-t11

	// Stage 4 (size 16)
	w1 := twiddle[1]
	w1 = complex(real(w1), -imag(w1))
	w3 := twiddle[3]
	w3 = complex(real(w3), -imag(w3))
	w5 := twiddle[5]
	w5 = complex(real(w5), -imag(w5))
	w7 := twiddle[7]
	w7 = complex(real(w7), -imag(w7))
	d0, d8 := c0+c8, c0-c8
	t1 = w1 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w2 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w3 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w4 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w5 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w6 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w7 * c15
	d7, d15 := c7+t7, c7-t7

	work = work[:n]
	work[0] = d0
	work[1] = d1
	work[2] = d2
	work[3] = d3
	work[4] = d4
	work[5] = d5
	work[6] = d6
	work[7] = d7
	work[8] = d8
	work[9] = d9
	work[10] = d10
	work[11] = d11
	work[12] = d12
	work[13] = d13
	work[14] = d14
	work[15] = d15

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
