package fft

import "math"

func forwardSixStepComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return sixStepForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseSixStepComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return sixStepInverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardSixStepComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return sixStepForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseSixStepComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return sixStepInverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func sixStepForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
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

	pairs := ComputeSquareTransposePairs(m)
	ApplyTransposePairs(data, pairs)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch, bitrev[:m]) {
			return false
		}
	}

	ApplyTransposePairs(data, pairs)

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

	ApplyTransposePairs(data, pairs)

	return true
}

func sixStepInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
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

	pairs := ComputeSquareTransposePairs(m)
	ApplyTransposePairs(data, pairs)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch, bitrev[:m]) {
			return false
		}
	}

	ApplyTransposePairs(data, pairs)

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

	ApplyTransposePairs(data, pairs)

	return true
}

func fillRowTwiddle[T Complex](rowTwiddle, twiddle []T, stride int) {
	for i := range rowTwiddle {
		rowTwiddle[i] = twiddle[i*stride]
	}
}

func intSqrt(n int) int {
	if n <= 0 {
		return 0
	}

	root := int(math.Sqrt(float64(n)))
	for (root+1)*(root+1) <= n {
		root++
	}

	for root*root > n {
		root--
	}

	return root
}
