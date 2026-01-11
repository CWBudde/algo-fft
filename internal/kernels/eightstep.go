package kernels

import (
	"github.com/MeKo-Christian/algo-fft/internal/math"
)

// ForwardEightStepComplex64 performs a forward eight-step FFT on complex64 data.
func ForwardEightStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return eightStepForward[complex64](dst, src, twiddle, scratch)
}

// InverseEightStepComplex64 performs an inverse eight-step FFT on complex64 data.
func InverseEightStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return eightStepInverse[complex64](dst, src, twiddle, scratch)
}

// ForwardEightStepComplex128 performs a forward eight-step FFT on complex128 data.
func ForwardEightStepComplex128(dst, src, twiddle, scratch []complex128) bool {
	return eightStepForward[complex128](dst, src, twiddle, scratch)
}

// InverseEightStepComplex128 performs an inverse eight-step FFT on complex128 data.
func InverseEightStepComplex128(dst, src, twiddle, scratch []complex128) bool {
	return eightStepInverse[complex128](dst, src, twiddle, scratch)
}

func eightStepForward[T Complex](dst, src, twiddle, scratch []T) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
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

	// Use shared transpose logic for correctness
	pairs := math.ComputeSquareTransposePairs(m)
	math.ApplyTransposePairs(data, pairs)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.ApplyTransposePairs(data, pairs)

	for i := range m {
		for j := range m {
			data[i*m+j] *= twiddle[(i*j)%n]
		}
	}

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.ApplyTransposePairs(data, pairs)

	return true
}

func eightStepInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
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

	pairs := math.ComputeSquareTransposePairs(m)
	math.ApplyTransposePairs(data, pairs)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.ApplyTransposePairs(data, pairs)

	for i := range m {
		for j := range m {
			data[i*m+j] *= conj(twiddle[(i*j)%n])
		}
	}

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.ApplyTransposePairs(data, pairs)

	return true
}
