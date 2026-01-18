package kernels

// forwardDIT8Radix8Complex64 computes an 8-point forward FFT using a single
// radix-8 butterfly for complex64 data. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix8Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]

	// Pre-load twiddle factors.
	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	a0 := s[0] + s[4]
	a1 := s[0] - s[4]
	a2 := s[2] + s[6]
	a3 := s[2] - s[6]

	e0 := a0 + a2
	e2 := a0 - a2
	e1 := a1 + complex(imag(a3), -real(a3))
	e3 := a1 + complex(-imag(a3), real(a3))

	a0 = s[1] + s[5]
	a1 = s[1] - s[5]
	a2 = s[3] + s[7]
	a3 = s[3] - s[7]

	o0 := a0 + a2
	o2 := a0 - a2
	o1 := a1 + complex(imag(a3), -real(a3))
	o3 := a1 + complex(-imag(a3), real(a3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	work[0] = e0 + o0
	work[4] = e0 - o0

	t := w1 * o1
	work[1] = e1 + t
	work[5] = e1 - t

	t = w2 * o2
	work[2] = e2 + t
	work[6] = e2 - t

	t = w3 * o3
	work[3] = e3 + t
	work[7] = e3 - t

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8Radix8Complex64 computes an 8-point inverse FFT using a single
// radix-8 butterfly for complex64 data. Uses conjugated twiddle factors and
// applies 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
//
//nolint:funlen
func inverseDIT8Radix8Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]

	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	a0 := s[0] + s[4]
	a1 := s[0] - s[4]
	a2 := s[2] + s[6]
	a3 := s[2] - s[6]

	e0 := a0 + a2
	e2 := a0 - a2
	e1 := a1 + complex(-imag(a3), real(a3))
	e3 := a1 + complex(imag(a3), -real(a3))

	a0 = s[1] + s[5]
	a1 = s[1] - s[5]
	a2 = s[3] + s[7]
	a3 = s[3] - s[7]

	o0 := a0 + a2
	o2 := a0 - a2
	o1 := a1 + complex(-imag(a3), real(a3))
	o3 := a1 + complex(imag(a3), -real(a3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	work[0] = e0 + o0
	work[4] = e0 - o0

	t := w1 * o1
	work[1] = e1 + t
	work[5] = e1 - t

	t = w2 * o2
	work[2] = e2 + t
	work[6] = e2 - t

	t = w3 * o3
	work[3] = e3 + t
	work[7] = e3 - t

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT8Radix8Complex128 computes an 8-point forward FFT using a single
// radix-8 butterfly for complex128 data. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT8Radix8Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]

	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]

	a0 := s[0] + s[4]
	a1 := s[0] - s[4]
	a2 := s[2] + s[6]
	a3 := s[2] - s[6]

	e0 := a0 + a2
	e2 := a0 - a2
	e1 := a1 + complex(imag(a3), -real(a3))
	e3 := a1 + complex(-imag(a3), real(a3))

	a0 = s[1] + s[5]
	a1 = s[1] - s[5]
	a2 = s[3] + s[7]
	a3 = s[3] - s[7]

	o0 := a0 + a2
	o2 := a0 - a2
	o1 := a1 + complex(imag(a3), -real(a3))
	o3 := a1 + complex(-imag(a3), real(a3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	work[0] = e0 + o0
	work[4] = e0 - o0

	t := w1 * o1
	work[1] = e1 + t
	work[5] = e1 - t

	t = w2 * o2
	work[2] = e2 + t
	work[6] = e2 - t

	t = w3 * o3
	work[3] = e3 + t
	work[7] = e3 - t

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8Radix8Complex128 computes an 8-point inverse FFT using a single
// radix-8 butterfly for complex128 data. Uses conjugated twiddle factors and
// applies 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
//
//nolint:funlen
func inverseDIT8Radix8Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 8

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]

	w1, w2, w3 := twiddle[1], twiddle[2], twiddle[3]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))

	a0 := s[0] + s[4]
	a1 := s[0] - s[4]
	a2 := s[2] + s[6]
	a3 := s[2] - s[6]

	e0 := a0 + a2
	e2 := a0 - a2
	e1 := a1 + complex(-imag(a3), real(a3))
	e3 := a1 + complex(imag(a3), -real(a3))

	a0 = s[1] + s[5]
	a1 = s[1] - s[5]
	a2 = s[3] + s[7]
	a3 = s[3] - s[7]

	o0 := a0 + a2
	o2 := a0 - a2
	o1 := a1 + complex(-imag(a3), real(a3))
	o3 := a1 + complex(imag(a3), -real(a3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	work[0] = e0 + o0
	work[4] = e0 - o0

	t := w1 * o1
	work[1] = e1 + t
	work[5] = e1 - t

	t = w2 * o2
	work[2] = e2 + t
	work[6] = e2 - t

	t = w3 * o3
	work[3] = e3 + t
	work[7] = e3 - t

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
