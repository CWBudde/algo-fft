package kernels

import (
	stdmath "math"

	"github.com/MeKo-Christian/algo-fft/internal/math"
)

// ForwardSixStepComplex64 performs a forward six-step FFT on complex64 data.
func ForwardSixStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return sixStepForward[complex64](dst, src, twiddle, scratch)
}

// InverseSixStepComplex64 performs an inverse six-step FFT on complex64 data.
func InverseSixStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return sixStepInverse[complex64](dst, src, twiddle, scratch)
}

// ForwardSixStepComplex128 performs a forward six-step FFT on complex128 data.
func ForwardSixStepComplex128(dst, src, twiddle, scratch []complex128) bool {
	return sixStepForward[complex128](dst, src, twiddle, scratch)
}

// InverseSixStepComplex128 performs an inverse six-step FFT on complex128 data.
func InverseSixStepComplex128(dst, src, twiddle, scratch []complex128) bool {
	return sixStepInverse[complex128](dst, src, twiddle, scratch)
}

func sixStepForward[T Complex](dst, src, twiddle, scratch []T) bool {
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

func sixStepInverse[T Complex](dst, src, twiddle, scratch []T) bool {
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

func fillRowTwiddle[T Complex](rowTwiddle, twiddle []T, stride int) {
	for i := range rowTwiddle {
		rowTwiddle[i] = twiddle[i*stride]
	}
}

func intSqrt(n int) int {
	if n <= 0 {
		return 0
	}

	root := int(stdmath.Sqrt(float64(n)))
	for (root+1)*(root+1) <= n {
		root++
	}

	for root*root > n {
		root--
	}

	return root
}
