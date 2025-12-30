package kernels

import "math"

func forwardRadix3Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix3Forward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix3Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix3Inverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardRadix3Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix3Forward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix3Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix3Inverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func radix3Forward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix3Transform(dst, src, twiddle, scratch, bitrev, false)
}

func radix3Inverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix3Transform(dst, src, twiddle, scratch, bitrev, true)
}

func radix3Transform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
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

	if !isPowerOf3(n) {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	digits := logBase3(n)
	for i := range n {
		work[i] = src[reverseBase3(i, digits)]
	}

	for size := 3; size <= n; size *= 3 {
		third := size / 3

		step := n / size
		for base := 0; base < n; base += size {
			for j := range third {
				idx0 := base + j
				idx1 := idx0 + third
				idx2 := idx1 + third

				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]

				if inverse {
					w1 = conj(w1)
					w2 = conj(w2)
				}

				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]

				var y0, y1, y2 T
				if inverse {
					y0, y1, y2 = butterfly3Inverse(a0, a1, a2)
				} else {
					y0, y1, y2 = butterfly3Forward(a0, a1, a2)
				}

				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	if inverse {
		scale := complexFromFloat64[T](1.0/float64(n), 0)
		for i := range dst {
			dst[i] *= scale
		}
	}

	return true
}

func butterfly3Forward[T Complex](a0, a1, a2 T) (T, T, T) {
	t1 := a1 + a2
	t2 := a1 - a2

	half := complexFromFloat64[T](-0.5, 0)
	coef := complexFromFloat64[T](0, -math.Sqrt(3)/2)

	y0 := a0 + t1
	base := a0 + half*t1
	y1 := base + coef*t2
	y2 := base - coef*t2

	return y0, y1, y2
}

func butterfly3Inverse[T Complex](a0, a1, a2 T) (T, T, T) {
	t1 := a1 + a2
	t2 := a1 - a2

	half := complexFromFloat64[T](-0.5, 0)
	coef := complexFromFloat64[T](0, math.Sqrt(3)/2)

	y0 := a0 + t1
	base := a0 + half*t1
	y1 := base + coef*t2
	y2 := base - coef*t2

	return y0, y1, y2
}

func reverseBase3(x, digits int) int {
	result := 0
	for range digits {
		result = result*3 + (x % 3)
		x /= 3
	}

	return result
}

func logBase3(n int) int {
	result := 0

	for n > 1 {
		n /= 3
		result++
	}

	return result
}
