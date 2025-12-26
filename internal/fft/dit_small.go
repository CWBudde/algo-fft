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

func forwardDIT64Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex64
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !forwardDIT32Complex64(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !forwardDIT32Complex64(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		t := twiddle[k] * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	return true
}

func inverseDIT64Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex64
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !inverseDIT32Complex64(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !inverseDIT32Complex64(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		tw := twiddle[k]
		tw = complex(real(tw), -imag(tw))
		t := tw * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	scale := complex(float32(0.5), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT64Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex128
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !forwardDIT32Complex128(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !forwardDIT32Complex128(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		t := twiddle[k] * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	return true
}

func inverseDIT64Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex128
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !inverseDIT32Complex128(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !inverseDIT32Complex128(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		tw := twiddle[k]
		tw = complex(real(tw), -imag(tw))
		t := tw * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	scale := complex(0.5, 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT128Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for j := range 64 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT128Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for j := range 64 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT128Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for j := range 64 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT128Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for j := range 64 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT32Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 32

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
	x16 := s[br[16]]
	x17 := s[br[17]]
	x18 := s[br[18]]
	x19 := s[br[19]]
	x20 := s[br[20]]
	x21 := s[br[21]]
	x22 := s[br[22]]
	x23 := s[br[23]]
	x24 := s[br[24]]
	x25 := s[br[25]]
	x26 := s[br[26]]
	x27 := s[br[27]]
	x28 := s[br[28]]
	x29 := s[br[29]]
	x30 := s[br[30]]
	x31 := s[br[31]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15
	a16, a17 := x16+x17, x16-x17
	a18, a19 := x18+x19, x18-x19
	a20, a21 := x20+x21, x20-x21
	a22, a23 := x22+x23, x22-x23
	a24, a25 := x24+x25, x24-x25
	a26, a27 := x26+x27, x26-x27
	a28, a29 := x28+x29, x28-x29
	a30, a31 := x30+x31, x30-x31

	// Stage 2 (size 4)
	w8 := twiddle[8]
	b0, b2 := a0+a2, a0-a2
	t1 := w8 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w8 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w8 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w8 * a15
	b13, b15 := a13+t13, a13-t13
	b16, b18 := a16+a18, a16-a18
	t17 := w8 * a19
	b17, b19 := a17+t17, a17-t17
	b20, b22 := a20+a22, a20-a22
	t21 := w8 * a23
	b21, b23 := a21+t21, a21-t21
	b24, b26 := a24+a26, a24-a26
	t25 := w8 * a27
	b25, b27 := a25+t25, a25-t25
	b28, b30 := a28+a30, a28-a30
	t29 := w8 * a31
	b29, b31 := a29+t29, a29-t29

	// Stage 3 (size 8)
	w4 := twiddle[4]
	w12 := twiddle[12]
	c0, c4 := b0+b4, b0-b4
	t1 = w4 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w8 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w12 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w4 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w8 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w12 * b15
	c11, c15 := b11+t11, b11-t11
	c16, c20 := b16+b20, b16-b20
	t17 = w4 * b21
	c17, c21 := b17+t17, b17-t17
	t18 := w8 * b22
	c18, c22 := b18+t18, b18-t18
	t19 := w12 * b23
	c19, c23 := b19+t19, b19-t19
	c24, c28 := b24+b28, b24-b28
	t25 = w4 * b29
	c25, c29 := b25+t25, b25-t25
	t26 := w8 * b30
	c26, c30 := b26+t26, b26-t26
	t27 := w12 * b31
	c27, c31 := b27+t27, b27-t27

	// Stage 4 (size 16)
	w2 := twiddle[2]
	w6 := twiddle[6]
	w10 := twiddle[10]
	w14 := twiddle[14]
	d0, d8 := c0+c8, c0-c8
	t1 = w2 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w4 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w6 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w8 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w10 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w12 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w14 * c15
	d7, d15 := c7+t7, c7-t7
	d16, d24 := c16+c24, c16-c24
	t17 = w2 * c25
	d17, d25 := c17+t17, c17-t17
	t18 = w4 * c26
	d18, d26 := c18+t18, c18-t18
	t19 = w6 * c27
	d19, d27 := c19+t19, c19-t19
	t20 := w8 * c28
	d20, d28 := c20+t20, c20-t20
	t21 = w10 * c29
	d21, d29 := c21+t21, c21-t21
	t22 := w12 * c30
	d22, d30 := c22+t22, c22-t22
	t23 := w14 * c31
	d23, d31 := c23+t23, c23-t23

	// Stage 5 (size 32)
	w1 := twiddle[1]
	w3 := twiddle[3]
	w5 := twiddle[5]
	w7 := twiddle[7]
	w9 := twiddle[9]
	w11 := twiddle[11]
	w13 := twiddle[13]
	w15 := twiddle[15]
	e0, e16 := d0+d16, d0-d16
	t1 = w1 * d17
	e1, e17 := d1+t1, d1-t1
	t2 = w2 * d18
	e2, e18 := d2+t2, d2-t2
	t3 = w3 * d19
	e3, e19 := d3+t3, d3-t3
	t4 = w4 * d20
	e4, e20 := d4+t4, d4-t4
	t5 = w5 * d21
	e5, e21 := d5+t5, d5-t5
	t6 = w6 * d22
	e6, e22 := d6+t6, d6-t6
	t7 = w7 * d23
	e7, e23 := d7+t7, d7-t7
	t8 := w8 * d24
	e8, e24 := d8+t8, d8-t8
	t9 = w9 * d25
	e9, e25 := d9+t9, d9-t9
	t10 = w10 * d26
	e10, e26 := d10+t10, d10-t10
	t11 = w11 * d27
	e11, e27 := d11+t11, d11-t11
	t12 := w12 * d28
	e12, e28 := d12+t12, d12-t12
	t13 = w13 * d29
	e13, e29 := d13+t13, d13-t13
	t14 := w14 * d30
	e14, e30 := d14+t14, d14-t14
	t15 := w15 * d31
	e15, e31 := d15+t15, d15-t15

	work = work[:n]
	work[0] = e0
	work[1] = e1
	work[2] = e2
	work[3] = e3
	work[4] = e4
	work[5] = e5
	work[6] = e6
	work[7] = e7
	work[8] = e8
	work[9] = e9
	work[10] = e10
	work[11] = e11
	work[12] = e12
	work[13] = e13
	work[14] = e14
	work[15] = e15
	work[16] = e16
	work[17] = e17
	work[18] = e18
	work[19] = e19
	work[20] = e20
	work[21] = e21
	work[22] = e22
	work[23] = e23
	work[24] = e24
	work[25] = e25
	work[26] = e26
	work[27] = e27
	work[28] = e28
	work[29] = e29
	work[30] = e30
	work[31] = e31

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT32Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 32

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
	x16 := s[br[16]]
	x17 := s[br[17]]
	x18 := s[br[18]]
	x19 := s[br[19]]
	x20 := s[br[20]]
	x21 := s[br[21]]
	x22 := s[br[22]]
	x23 := s[br[23]]
	x24 := s[br[24]]
	x25 := s[br[25]]
	x26 := s[br[26]]
	x27 := s[br[27]]
	x28 := s[br[28]]
	x29 := s[br[29]]
	x30 := s[br[30]]
	x31 := s[br[31]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15
	a16, a17 := x16+x17, x16-x17
	a18, a19 := x18+x19, x18-x19
	a20, a21 := x20+x21, x20-x21
	a22, a23 := x22+x23, x22-x23
	a24, a25 := x24+x25, x24-x25
	a26, a27 := x26+x27, x26-x27
	a28, a29 := x28+x29, x28-x29
	a30, a31 := x30+x31, x30-x31

	// Stage 2 (size 4)
	w8 := twiddle[8]
	w8 = complex(real(w8), -imag(w8))
	b0, b2 := a0+a2, a0-a2
	t1 := w8 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w8 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w8 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w8 * a15
	b13, b15 := a13+t13, a13-t13
	b16, b18 := a16+a18, a16-a18
	t17 := w8 * a19
	b17, b19 := a17+t17, a17-t17
	b20, b22 := a20+a22, a20-a22
	t21 := w8 * a23
	b21, b23 := a21+t21, a21-t21
	b24, b26 := a24+a26, a24-a26
	t25 := w8 * a27
	b25, b27 := a25+t25, a25-t25
	b28, b30 := a28+a30, a28-a30
	t29 := w8 * a31
	b29, b31 := a29+t29, a29-t29

	// Stage 3 (size 8)
	w4 := twiddle[4]
	w4 = complex(real(w4), -imag(w4))
	w12 := twiddle[12]
	w12 = complex(real(w12), -imag(w12))
	c0, c4 := b0+b4, b0-b4
	t1 = w4 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w8 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w12 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w4 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w8 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w12 * b15
	c11, c15 := b11+t11, b11-t11
	c16, c20 := b16+b20, b16-b20
	t17 = w4 * b21
	c17, c21 := b17+t17, b17-t17
	t18 := w8 * b22
	c18, c22 := b18+t18, b18-t18
	t19 := w12 * b23
	c19, c23 := b19+t19, b19-t19
	c24, c28 := b24+b28, b24-b28
	t25 = w4 * b29
	c25, c29 := b25+t25, b25-t25
	t26 := w8 * b30
	c26, c30 := b26+t26, b26-t26
	t27 := w12 * b31
	c27, c31 := b27+t27, b27-t27

	// Stage 4 (size 16)
	w2 := twiddle[2]
	w2 = complex(real(w2), -imag(w2))
	w6 := twiddle[6]
	w6 = complex(real(w6), -imag(w6))
	w10 := twiddle[10]
	w10 = complex(real(w10), -imag(w10))
	w14 := twiddle[14]
	w14 = complex(real(w14), -imag(w14))
	d0, d8 := c0+c8, c0-c8
	t1 = w2 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w4 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w6 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w8 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w10 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w12 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w14 * c15
	d7, d15 := c7+t7, c7-t7
	d16, d24 := c16+c24, c16-c24
	t17 = w2 * c25
	d17, d25 := c17+t17, c17-t17
	t18 = w4 * c26
	d18, d26 := c18+t18, c18-t18
	t19 = w6 * c27
	d19, d27 := c19+t19, c19-t19
	t20 := w8 * c28
	d20, d28 := c20+t20, c20-t20
	t21 = w10 * c29
	d21, d29 := c21+t21, c21-t21
	t22 := w12 * c30
	d22, d30 := c22+t22, c22-t22
	t23 := w14 * c31
	d23, d31 := c23+t23, c23-t23

	// Stage 5 (size 32)
	w1 := twiddle[1]
	w1 = complex(real(w1), -imag(w1))
	w3 := twiddle[3]
	w3 = complex(real(w3), -imag(w3))
	w5 := twiddle[5]
	w5 = complex(real(w5), -imag(w5))
	w7 := twiddle[7]
	w7 = complex(real(w7), -imag(w7))
	w9 := twiddle[9]
	w9 = complex(real(w9), -imag(w9))
	w11 := twiddle[11]
	w11 = complex(real(w11), -imag(w11))
	w13 := twiddle[13]
	w13 = complex(real(w13), -imag(w13))
	w15 := twiddle[15]
	w15 = complex(real(w15), -imag(w15))
	e0, e16 := d0+d16, d0-d16
	t1 = w1 * d17
	e1, e17 := d1+t1, d1-t1
	t2 = w2 * d18
	e2, e18 := d2+t2, d2-t2
	t3 = w3 * d19
	e3, e19 := d3+t3, d3-t3
	t4 = w4 * d20
	e4, e20 := d4+t4, d4-t4
	t5 = w5 * d21
	e5, e21 := d5+t5, d5-t5
	t6 = w6 * d22
	e6, e22 := d6+t6, d6-t6
	t7 = w7 * d23
	e7, e23 := d7+t7, d7-t7
	t8 := w8 * d24
	e8, e24 := d8+t8, d8-t8
	t9 = w9 * d25
	e9, e25 := d9+t9, d9-t9
	t10 = w10 * d26
	e10, e26 := d10+t10, d10-t10
	t11 = w11 * d27
	e11, e27 := d11+t11, d11-t11
	t12 := w12 * d28
	e12, e28 := d12+t12, d12-t12
	t13 = w13 * d29
	e13, e29 := d13+t13, d13-t13
	t14 := w14 * d30
	e14, e30 := d14+t14, d14-t14
	t15 := w15 * d31
	e15, e31 := d15+t15, d15-t15

	work = work[:n]
	work[0] = e0
	work[1] = e1
	work[2] = e2
	work[3] = e3
	work[4] = e4
	work[5] = e5
	work[6] = e6
	work[7] = e7
	work[8] = e8
	work[9] = e9
	work[10] = e10
	work[11] = e11
	work[12] = e12
	work[13] = e13
	work[14] = e14
	work[15] = e15
	work[16] = e16
	work[17] = e17
	work[18] = e18
	work[19] = e19
	work[20] = e20
	work[21] = e21
	work[22] = e22
	work[23] = e23
	work[24] = e24
	work[25] = e25
	work[26] = e26
	work[27] = e27
	work[28] = e28
	work[29] = e29
	work[30] = e30
	work[31] = e31

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT32Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 32

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
	x16 := s[br[16]]
	x17 := s[br[17]]
	x18 := s[br[18]]
	x19 := s[br[19]]
	x20 := s[br[20]]
	x21 := s[br[21]]
	x22 := s[br[22]]
	x23 := s[br[23]]
	x24 := s[br[24]]
	x25 := s[br[25]]
	x26 := s[br[26]]
	x27 := s[br[27]]
	x28 := s[br[28]]
	x29 := s[br[29]]
	x30 := s[br[30]]
	x31 := s[br[31]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15
	a16, a17 := x16+x17, x16-x17
	a18, a19 := x18+x19, x18-x19
	a20, a21 := x20+x21, x20-x21
	a22, a23 := x22+x23, x22-x23
	a24, a25 := x24+x25, x24-x25
	a26, a27 := x26+x27, x26-x27
	a28, a29 := x28+x29, x28-x29
	a30, a31 := x30+x31, x30-x31

	// Stage 2 (size 4)
	w8 := twiddle[8]
	b0, b2 := a0+a2, a0-a2
	t1 := w8 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w8 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w8 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w8 * a15
	b13, b15 := a13+t13, a13-t13
	b16, b18 := a16+a18, a16-a18
	t17 := w8 * a19
	b17, b19 := a17+t17, a17-t17
	b20, b22 := a20+a22, a20-a22
	t21 := w8 * a23
	b21, b23 := a21+t21, a21-t21
	b24, b26 := a24+a26, a24-a26
	t25 := w8 * a27
	b25, b27 := a25+t25, a25-t25
	b28, b30 := a28+a30, a28-a30
	t29 := w8 * a31
	b29, b31 := a29+t29, a29-t29

	// Stage 3 (size 8)
	w4 := twiddle[4]
	w12 := twiddle[12]
	c0, c4 := b0+b4, b0-b4
	t1 = w4 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w8 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w12 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w4 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w8 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w12 * b15
	c11, c15 := b11+t11, b11-t11
	c16, c20 := b16+b20, b16-b20
	t17 = w4 * b21
	c17, c21 := b17+t17, b17-t17
	t18 := w8 * b22
	c18, c22 := b18+t18, b18-t18
	t19 := w12 * b23
	c19, c23 := b19+t19, b19-t19
	c24, c28 := b24+b28, b24-b28
	t25 = w4 * b29
	c25, c29 := b25+t25, b25-t25
	t26 := w8 * b30
	c26, c30 := b26+t26, b26-t26
	t27 := w12 * b31
	c27, c31 := b27+t27, b27-t27

	// Stage 4 (size 16)
	w2 := twiddle[2]
	w6 := twiddle[6]
	w10 := twiddle[10]
	w14 := twiddle[14]
	d0, d8 := c0+c8, c0-c8
	t1 = w2 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w4 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w6 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w8 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w10 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w12 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w14 * c15
	d7, d15 := c7+t7, c7-t7
	d16, d24 := c16+c24, c16-c24
	t17 = w2 * c25
	d17, d25 := c17+t17, c17-t17
	t18 = w4 * c26
	d18, d26 := c18+t18, c18-t18
	t19 = w6 * c27
	d19, d27 := c19+t19, c19-t19
	t20 := w8 * c28
	d20, d28 := c20+t20, c20-t20
	t21 = w10 * c29
	d21, d29 := c21+t21, c21-t21
	t22 := w12 * c30
	d22, d30 := c22+t22, c22-t22
	t23 := w14 * c31
	d23, d31 := c23+t23, c23-t23

	// Stage 5 (size 32)
	w1 := twiddle[1]
	w3 := twiddle[3]
	w5 := twiddle[5]
	w7 := twiddle[7]
	w9 := twiddle[9]
	w11 := twiddle[11]
	w13 := twiddle[13]
	w15 := twiddle[15]
	e0, e16 := d0+d16, d0-d16
	t1 = w1 * d17
	e1, e17 := d1+t1, d1-t1
	t2 = w2 * d18
	e2, e18 := d2+t2, d2-t2
	t3 = w3 * d19
	e3, e19 := d3+t3, d3-t3
	t4 = w4 * d20
	e4, e20 := d4+t4, d4-t4
	t5 = w5 * d21
	e5, e21 := d5+t5, d5-t5
	t6 = w6 * d22
	e6, e22 := d6+t6, d6-t6
	t7 = w7 * d23
	e7, e23 := d7+t7, d7-t7
	t8 := w8 * d24
	e8, e24 := d8+t8, d8-t8
	t9 = w9 * d25
	e9, e25 := d9+t9, d9-t9
	t10 = w10 * d26
	e10, e26 := d10+t10, d10-t10
	t11 = w11 * d27
	e11, e27 := d11+t11, d11-t11
	t12 := w12 * d28
	e12, e28 := d12+t12, d12-t12
	t13 = w13 * d29
	e13, e29 := d13+t13, d13-t13
	t14 := w14 * d30
	e14, e30 := d14+t14, d14-t14
	t15 := w15 * d31
	e15, e31 := d15+t15, d15-t15

	work = work[:n]
	work[0] = e0
	work[1] = e1
	work[2] = e2
	work[3] = e3
	work[4] = e4
	work[5] = e5
	work[6] = e6
	work[7] = e7
	work[8] = e8
	work[9] = e9
	work[10] = e10
	work[11] = e11
	work[12] = e12
	work[13] = e13
	work[14] = e14
	work[15] = e15
	work[16] = e16
	work[17] = e17
	work[18] = e18
	work[19] = e19
	work[20] = e20
	work[21] = e21
	work[22] = e22
	work[23] = e23
	work[24] = e24
	work[25] = e25
	work[26] = e26
	work[27] = e27
	work[28] = e28
	work[29] = e29
	work[30] = e30
	work[31] = e31

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT32Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 32

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
	x16 := s[br[16]]
	x17 := s[br[17]]
	x18 := s[br[18]]
	x19 := s[br[19]]
	x20 := s[br[20]]
	x21 := s[br[21]]
	x22 := s[br[22]]
	x23 := s[br[23]]
	x24 := s[br[24]]
	x25 := s[br[25]]
	x26 := s[br[26]]
	x27 := s[br[27]]
	x28 := s[br[28]]
	x29 := s[br[29]]
	x30 := s[br[30]]
	x31 := s[br[31]]

	// Stage 1 (size 2)
	a0, a1 := x0+x1, x0-x1
	a2, a3 := x2+x3, x2-x3
	a4, a5 := x4+x5, x4-x5
	a6, a7 := x6+x7, x6-x7
	a8, a9 := x8+x9, x8-x9
	a10, a11 := x10+x11, x10-x11
	a12, a13 := x12+x13, x12-x13
	a14, a15 := x14+x15, x14-x15
	a16, a17 := x16+x17, x16-x17
	a18, a19 := x18+x19, x18-x19
	a20, a21 := x20+x21, x20-x21
	a22, a23 := x22+x23, x22-x23
	a24, a25 := x24+x25, x24-x25
	a26, a27 := x26+x27, x26-x27
	a28, a29 := x28+x29, x28-x29
	a30, a31 := x30+x31, x30-x31

	// Stage 2 (size 4)
	w8 := twiddle[8]
	w8 = complex(real(w8), -imag(w8))
	b0, b2 := a0+a2, a0-a2
	t1 := w8 * a3
	b1, b3 := a1+t1, a1-t1
	b4, b6 := a4+a6, a4-a6
	t5 := w8 * a7
	b5, b7 := a5+t5, a5-t5
	b8, b10 := a8+a10, a8-a10
	t9 := w8 * a11
	b9, b11 := a9+t9, a9-t9
	b12, b14 := a12+a14, a12-a14
	t13 := w8 * a15
	b13, b15 := a13+t13, a13-t13
	b16, b18 := a16+a18, a16-a18
	t17 := w8 * a19
	b17, b19 := a17+t17, a17-t17
	b20, b22 := a20+a22, a20-a22
	t21 := w8 * a23
	b21, b23 := a21+t21, a21-t21
	b24, b26 := a24+a26, a24-a26
	t25 := w8 * a27
	b25, b27 := a25+t25, a25-t25
	b28, b30 := a28+a30, a28-a30
	t29 := w8 * a31
	b29, b31 := a29+t29, a29-t29

	// Stage 3 (size 8)
	w4 := twiddle[4]
	w4 = complex(real(w4), -imag(w4))
	w12 := twiddle[12]
	w12 = complex(real(w12), -imag(w12))
	c0, c4 := b0+b4, b0-b4
	t1 = w4 * b5
	c1, c5 := b1+t1, b1-t1
	t2 := w8 * b6
	c2, c6 := b2+t2, b2-t2
	t3 := w12 * b7
	c3, c7 := b3+t3, b3-t3
	c8, c12 := b8+b12, b8-b12
	t9 = w4 * b13
	c9, c13 := b9+t9, b9-t9
	t10 := w8 * b14
	c10, c14 := b10+t10, b10-t10
	t11 := w12 * b15
	c11, c15 := b11+t11, b11-t11
	c16, c20 := b16+b20, b16-b20
	t17 = w4 * b21
	c17, c21 := b17+t17, b17-t17
	t18 := w8 * b22
	c18, c22 := b18+t18, b18-t18
	t19 := w12 * b23
	c19, c23 := b19+t19, b19-t19
	c24, c28 := b24+b28, b24-b28
	t25 = w4 * b29
	c25, c29 := b25+t25, b25-t25
	t26 := w8 * b30
	c26, c30 := b26+t26, b26-t26
	t27 := w12 * b31
	c27, c31 := b27+t27, b27-t27

	// Stage 4 (size 16)
	w2 := twiddle[2]
	w2 = complex(real(w2), -imag(w2))
	w6 := twiddle[6]
	w6 = complex(real(w6), -imag(w6))
	w10 := twiddle[10]
	w10 = complex(real(w10), -imag(w10))
	w14 := twiddle[14]
	w14 = complex(real(w14), -imag(w14))
	d0, d8 := c0+c8, c0-c8
	t1 = w2 * c9
	d1, d9 := c1+t1, c1-t1
	t2 = w4 * c10
	d2, d10 := c2+t2, c2-t2
	t3 = w6 * c11
	d3, d11 := c3+t3, c3-t3
	t4 := w8 * c12
	d4, d12 := c4+t4, c4-t4
	t5 = w10 * c13
	d5, d13 := c5+t5, c5-t5
	t6 := w12 * c14
	d6, d14 := c6+t6, c6-t6
	t7 := w14 * c15
	d7, d15 := c7+t7, c7-t7
	d16, d24 := c16+c24, c16-c24
	t17 = w2 * c25
	d17, d25 := c17+t17, c17-t17
	t18 = w4 * c26
	d18, d26 := c18+t18, c18-t18
	t19 = w6 * c27
	d19, d27 := c19+t19, c19-t19
	t20 := w8 * c28
	d20, d28 := c20+t20, c20-t20
	t21 = w10 * c29
	d21, d29 := c21+t21, c21-t21
	t22 := w12 * c30
	d22, d30 := c22+t22, c22-t22
	t23 := w14 * c31
	d23, d31 := c23+t23, c23-t23

	// Stage 5 (size 32)
	w1 := twiddle[1]
	w1 = complex(real(w1), -imag(w1))
	w3 := twiddle[3]
	w3 = complex(real(w3), -imag(w3))
	w5 := twiddle[5]
	w5 = complex(real(w5), -imag(w5))
	w7 := twiddle[7]
	w7 = complex(real(w7), -imag(w7))
	w9 := twiddle[9]
	w9 = complex(real(w9), -imag(w9))
	w11 := twiddle[11]
	w11 = complex(real(w11), -imag(w11))
	w13 := twiddle[13]
	w13 = complex(real(w13), -imag(w13))
	w15 := twiddle[15]
	w15 = complex(real(w15), -imag(w15))
	e0, e16 := d0+d16, d0-d16
	t1 = w1 * d17
	e1, e17 := d1+t1, d1-t1
	t2 = w2 * d18
	e2, e18 := d2+t2, d2-t2
	t3 = w3 * d19
	e3, e19 := d3+t3, d3-t3
	t4 = w4 * d20
	e4, e20 := d4+t4, d4-t4
	t5 = w5 * d21
	e5, e21 := d5+t5, d5-t5
	t6 = w6 * d22
	e6, e22 := d6+t6, d6-t6
	t7 = w7 * d23
	e7, e23 := d7+t7, d7-t7
	t8 := w8 * d24
	e8, e24 := d8+t8, d8-t8
	t9 = w9 * d25
	e9, e25 := d9+t9, d9-t9
	t10 = w10 * d26
	e10, e26 := d10+t10, d10-t10
	t11 = w11 * d27
	e11, e27 := d11+t11, d11-t11
	t12 := w12 * d28
	e12, e28 := d12+t12, d12-t12
	t13 = w13 * d29
	e13, e29 := d13+t13, d13-t13
	t14 := w14 * d30
	e14, e30 := d14+t14, d14-t14
	t15 := w15 * d31
	e15, e31 := d15+t15, d15-t15

	work = work[:n]
	work[0] = e0
	work[1] = e1
	work[2] = e2
	work[3] = e3
	work[4] = e4
	work[5] = e5
	work[6] = e6
	work[7] = e7
	work[8] = e8
	work[9] = e9
	work[10] = e10
	work[11] = e11
	work[12] = e12
	work[13] = e13
	work[14] = e14
	work[15] = e15
	work[16] = e16
	work[17] = e17
	work[18] = e18
	work[19] = e19
	work[20] = e20
	work[21] = e21
	work[22] = e22
	work[23] = e23
	work[24] = e24
	work[25] = e25
	work[26] = e26
	work[27] = e27
	work[28] = e28
	work[29] = e29
	work[30] = e30
	work[31] = e31

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
