package kernels

// ForwardEightStepComplex64 performs a forward eight-step FFT on complex64 data.
func ForwardEightStepComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return eightStepForward[complex64](dst, src, twiddle, scratch, bitrev)
}

// InverseEightStepComplex64 performs an inverse eight-step FFT on complex64 data.
func InverseEightStepComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return eightStepInverse[complex64](dst, src, twiddle, scratch, bitrev)
}

// ForwardEightStepComplex128 performs a forward eight-step FFT on complex128 data.
func ForwardEightStepComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return eightStepForward[complex128](dst, src, twiddle, scratch, bitrev)
}

// InverseEightStepComplex128 performs an inverse eight-step FFT on complex128 data.
func InverseEightStepComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return eightStepInverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func eightStepForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
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

	m := intSqrt(n)
	if m*m != n {
		return false
	}

	if sameSlice(dst, src) {
		copy(scratch, src)
		src = scratch
	}

	data := dst
	if !sameSlice(dst, src) {
		copy(dst, src)
	}

	block := eightStepBlockSize(m)
	transposeSquareBlocked(data, m, block)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch, bitrev[:m]) {
			return false
		}
	}

	transposeSquareBlocked(data, m, block)

	for i := range m {
		for j := range m {
			data[i*m+j] *= twiddle[(i*j)%n]
		}
	}

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch, bitrev[:m]) {
			return false
		}
	}

	transposeSquareBlocked(data, m, block)

	return true
}

func eightStepInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
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

	m := intSqrt(n)
	if m*m != n {
		return false
	}

	if sameSlice(dst, src) {
		copy(scratch, src)
		src = scratch
	}

	data := dst
	if !sameSlice(dst, src) {
		copy(dst, src)
	}

	block := eightStepBlockSize(m)
	transposeSquareBlocked(data, m, block)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch, bitrev[:m]) {
			return false
		}
	}

	transposeSquareBlocked(data, m, block)

	for i := range m {
		for j := range m {
			data[i*m+j] *= conj(twiddle[(i*j)%n])
		}
	}

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch, bitrev[:m]) {
			return false
		}
	}

	transposeSquareBlocked(data, m, block)

	return true
}

func transposeSquareBlocked[T any](data []T, n, block int) {
	if n <= 1 {
		return
	}

	if block <= 0 || block > n {
		block = n
	}

	for i := 0; i < n; i += block {
		imax := i + block
		if imax > n {
			imax = n
		}

		for j := i + 1; j < n; j += block {
			jmax := j + block
			if jmax > n {
				jmax = n
			}

			for r := i; r < imax; r++ {
				for c := j; c < jmax; c++ {
					a := r*n + c
					b := c*n + r
					data[a], data[b] = data[b], data[a]
				}
			}
		}
	}
}

func eightStepBlockSize(n int) int {
	if n <= 32 {
		return n
	}

	if n <= 64 {
		return 32
	}

	return 64
}
