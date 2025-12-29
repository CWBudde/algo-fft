package fft

// forwardDIT4Radix4Complex64 computes a 4-point forward FFT using the
// radix-4 algorithm for complex64 data. For size 4, this is just a single
// radix-4 butterfly with no twiddle factors needed (all W^0 = 1).
// No bit-reversal needed for size 4 with radix-4!
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT4Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// For radix-4 size 4, input order is natural: [0, 1, 2, 3]
	// No bit-reversal needed!
	s := src[:n]

	// Single radix-4 butterfly (no twiddle factors needed)
	// x0, x1, x2, x3 -> y0, y1, y2, y3
	//
	// t0 = x0 + x2
	// t1 = x0 - x2
	// t2 = x1 + x3
	// t3 = x1 - x3
	//
	// y0 = t0 + t2
	// y1 = t1 + t3*(-i)    where -i multiply: (r,i) -> (i,-r)
	// y2 = t0 - t2
	// y3 = t1 - t3*(-i)

	x0, x1, x2, x3 := s[0], s[1], s[2], s[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	// Multiply t3 by -i: (r,i) * (-i) = (i,-r)
	t3NegI := complex(imag(t3), -real(t3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = t0 + t2
	work[1] = t1 + t3NegI
	work[2] = t0 - t2
	work[3] = t1 - t3NegI

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	return true
}

// inverseDIT4Radix4Complex64 computes a 4-point inverse FFT using the
// radix-4 algorithm for complex64 data. Uses +i instead of -i and applies
// 1/N scaling at the end.
// Returns false if any slice is too small.
func inverseDIT4Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]

	// Single radix-4 butterfly with +i instead of -i
	x0, x1, x2, x3 := s[0], s[1], s[2], s[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	// Multiply t3 by +i: (r,i) * (i) = (-i,r)
	t3PosI := complex(-imag(t3), real(t3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = t0 + t2
	work[1] = t1 + t3PosI
	work[2] = t0 - t2
	work[3] = t1 - t3PosI

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT4Radix4Complex128 computes a 4-point forward FFT using the
// radix-4 algorithm for complex128 data.
// Returns false if any slice is too small.
func forwardDIT4Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]

	x0, x1, x2, x3 := s[0], s[1], s[2], s[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	t3NegI := complex(imag(t3), -real(t3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = t0 + t2
	work[1] = t1 + t3NegI
	work[2] = t0 - t2
	work[3] = t1 - t3NegI

	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	return true
}

// inverseDIT4Radix4Complex128 computes a 4-point inverse FFT using the
// radix-4 algorithm for complex128 data.
// Returns false if any slice is too small.
func inverseDIT4Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]

	x0, x1, x2, x3 := s[0], s[1], s[2], s[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	t3PosI := complex(-imag(t3), real(t3))

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	work[0] = t0 + t2
	work[1] = t1 + t3PosI
	work[2] = t0 - t2
	work[3] = t1 - t3PosI

	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
