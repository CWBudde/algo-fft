package fft

import "math"

var (
	radix5Twiddles64  [4]complex64
	radix5Twiddles128 [4]complex128
)

func init() {
	for k := 1; k <= 4; k++ {
		angle := -2 * math.Pi * float64(k) / 5
		re := math.Cos(angle)
		im := math.Sin(angle)
		radix5Twiddles128[k-1] = complex(re, im)
		radix5Twiddles64[k-1] = complex(float32(re), float32(im))
	}
}

func forwardRadix5Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix5Forward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix5Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix5Inverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardRadix5Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix5Forward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix5Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix5Inverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func radix5Forward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix5Transform(dst, src, twiddle, scratch, bitrev, false)
}

func radix5Inverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix5Transform(dst, src, twiddle, scratch, bitrev, true)
}

func radix5Transform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
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

	if !isPowerOf5(n) {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	digits := logBase5(n)
	for i := range n {
		work[i] = src[reverseBase5(i, digits)]
	}

	for size := 5; size <= n; size *= 5 {
		span := size / 5

		step := n / size
		for base := 0; base < n; base += size {
			for j := range span {
				idx0 := base + j
				idx1 := idx0 + span
				idx2 := idx1 + span
				idx3 := idx2 + span
				idx4 := idx3 + span

				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]
				w3 := twiddle[3*j*step]
				w4 := twiddle[4*j*step]

				if inverse {
					w1 = conj(w1)
					w2 = conj(w2)
					w3 = conj(w3)
					w4 = conj(w4)
				}

				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]
				a3 := w3 * work[idx3]
				a4 := w4 * work[idx4]

				var y0, y1, y2, y3, y4 T
				if inverse {
					y0, y1, y2, y3, y4 = butterfly5Inverse(a0, a1, a2, a3, a4)
				} else {
					y0, y1, y2, y3, y4 = butterfly5Forward(a0, a1, a2, a3, a4)
				}

				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
				work[idx3] = y3
				work[idx4] = y4
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

func butterfly5Forward[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	w1, w2, w3, w4 := radix5Twiddles[T]()

	y0 := a0 + a1 + a2 + a3 + a4
	y1 := a0 + a1*w1 + a2*w2 + a3*w3 + a4*w4
	y2 := a0 + a1*w2 + a2*w4 + a3*w1 + a4*w3
	y3 := a0 + a1*w3 + a2*w1 + a3*w4 + a4*w2
	y4 := a0 + a1*w4 + a2*w3 + a3*w2 + a4*w1

	return y0, y1, y2, y3, y4
}

func butterfly5Inverse[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	w1, w2, w3, w4 := radix5Twiddles[T]()
	w1 = conj(w1)
	w2 = conj(w2)
	w3 = conj(w3)
	w4 = conj(w4)

	y0 := a0 + a1 + a2 + a3 + a4
	y1 := a0 + a1*w1 + a2*w2 + a3*w3 + a4*w4
	y2 := a0 + a1*w2 + a2*w4 + a3*w1 + a4*w3
	y3 := a0 + a1*w3 + a2*w1 + a3*w4 + a4*w2
	y4 := a0 + a1*w4 + a2*w3 + a3*w2 + a4*w1

	return y0, y1, y2, y3, y4
}

func radix5Twiddles[T Complex]() (T, T, T, T) {
	var zero T
	switch any(zero).(type) {
	case complex64:
		return any(radix5Twiddles64[0]).(T),
			any(radix5Twiddles64[1]).(T),
			any(radix5Twiddles64[2]).(T),
			any(radix5Twiddles64[3]).(T)
	case complex128:
		return any(radix5Twiddles128[0]).(T),
			any(radix5Twiddles128[1]).(T),
			any(radix5Twiddles128[2]).(T),
			any(radix5Twiddles128[3]).(T)
	default:
		panic("unsupported complex type")
	}
}

func reverseBase5(x, digits int) int {
	result := 0
	for range digits {
		result = result*5 + (x % 5)
		x /= 5
	}

	return result
}

func logBase5(n int) int {
	result := 0

	for n > 1 {
		n /= 5
		result++
	}

	return result
}
