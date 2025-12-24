package fft

func forwardDITComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return ditForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseDITComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return ditInverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardDITComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return ditForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseDITComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return ditInverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func ditForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1

		step := n / size
		for base := 0; base < n; base += size {
			for j := range half {
				index1 := base + j
				index2 := index1 + half
				tw := twiddle[j*step]
				t := tw * work[index2]
				u := work[index1]
				work[index1] = u + t
				work[index2] = u - t
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func ditInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1

		step := n / size
		for base := 0; base < n; base += size {
			for j := range half {
				index1 := base + j
				index2 := index1 + half
				tw := conj(twiddle[j*step])
				t := tw * work[index2]
				u := work[index1]
				work[index1] = u + t
				work[index2] = u - t
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complexFromFloat64[T](1.0/float64(n), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}
