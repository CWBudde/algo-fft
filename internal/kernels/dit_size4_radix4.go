package kernels

// forwardDIT4Radix4Complex64 computes a 4-point forward FFT using the
// radix-4 algorithm for complex64 data. For size 4, this is just a single
// radix-4 butterfly with no twiddle factors needed (all W^0 = 1).
// No bit-reversal needed for size 4 with radix-4!
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT4Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	inPlace := &dst[0] == &src[0]

	// Choose output buffer.
	work := dst[:n]
	if inPlace {
		work = scratch[:n]
	}

	x0, x1, x2, x3 := src[0], src[1], src[2], src[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	// (r,i) * (-i) = (i,-r)
	t3NegI := complex(imag(t3), -real(t3))

	work[0] = t0 + t2
	work[1] = t1 + t3NegI
	work[2] = t0 - t2
	work[3] = t1 - t3NegI

	if inPlace {
		copy(dst[:n], work)
	}
	return true
}

// inverseDIT4Radix4Complex64 computes a 4-point inverse FFT using the
// radix-4 algorithm for complex64 data. Uses +i instead of -i and applies
// 1/N scaling at the end.
// Returns false if any slice is too small.
func inverseDIT4Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 4
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	// One alias check.
	inPlace := &dst[0] == &src[0]

	// Choose output buffer once.
	work := dst[:n]
	if inPlace {
		work = scratch[:n]
	}

	x0, x1, x2, x3 := src[0], src[1], src[2], src[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	// Multiply t3 by +i: (r,i) * i = (-i, r)
	t3PosI := complex(-imag(t3), real(t3))

	// Fold 1/N scaling (N=4) into the stores.
	const s complex64 = 0.25
	work[0] = (t0 + t2) * s
	work[1] = (t1 + t3PosI) * s
	work[2] = (t0 - t2) * s
	work[3] = (t1 - t3PosI) * s

	if inPlace {
		copy(dst[:n], work)
	}
	return true
}

// forwardDIT4Radix4Complex128 computes a 4-point forward FFT using the
// radix-4 algorithm for complex128 data.
// Returns false if any slice is too small.
func forwardDIT4Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	inPlace := &dst[0] == &src[0]

	// Choose output buffer.
	work := dst[:n]
	if inPlace {
		work = scratch[:n]
	}

	x0, x1, x2, x3 := src[0], src[1], src[2], src[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	t3NegI := complex(imag(t3), -real(t3))

	work[0] = t0 + t2
	work[1] = t1 + t3NegI
	work[2] = t0 - t2
	work[3] = t1 - t3NegI

	if inPlace {
		copy(dst[:n], work)
	}

	return true
}

// inverseDIT4Radix4Complex128 computes a 4-point inverse FFT using the
// radix-4 algorithm for complex128 data. Uses +i instead of -i and applies
// 1/N scaling at the end.
// Returns false if any slice is too small.
func inverseDIT4Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 4

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// One alias check.
	inPlace := &dst[0] == &src[0]

	// Choose output buffer once.
	work := dst[:n]
	if inPlace {
		work = scratch[:n]
	}

	x0, x1, x2, x3 := src[0], src[1], src[2], src[3]

	t0 := x0 + x2
	t1 := x0 - x2
	t2 := x1 + x3
	t3 := x1 - x3

	// Multiply t3 by +i: (r,i) * i = (-i, r)
	t3PosI := complex(-imag(t3), real(t3))

	// Fold 1/N scaling (N=4) into the stores.
	const s complex128 = 0.25
	work[0] = (t0 + t2) * s
	work[1] = (t1 + t3PosI) * s
	work[2] = (t0 - t2) * s
	work[3] = (t1 - t3PosI) * s

	if inPlace {
		copy(dst[:n], work)
	}
	return true
}

// forwardDIT4Radix4Complex64 is the wrapper for the 4-point forward kernel.
