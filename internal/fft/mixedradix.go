package fft

const mixedRadixMaxStages = 64

func forwardMixedRadixComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return mixedRadixForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseMixedRadixComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return mixedRadixInverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardMixedRadixComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return mixedRadixForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseMixedRadixComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return mixedRadixInverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func mixedRadixForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, bitrev, false)
}

func mixedRadixInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, bitrev, true)
}

func mixedRadixTransform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
	_ = bitrev

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

	var radices [mixedRadixMaxStages]int

	stageCount := mixedRadixSchedule(n, &radices)
	if stageCount == 0 {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	mixedRadixRecursive(work, src, n, 1, 1, radices[:stageCount], twiddle, scratch, inverse)

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

func mixedRadixSchedule(n int, radices *[mixedRadixMaxStages]int) int {
	if n < 2 {
		return 0
	}

	count := 0

	for n > 1 {
		switch {
		case n%5 == 0:
			radices[count] = 5
			n /= 5
		case n%4 == 0:
			radices[count] = 4
			n /= 4
		case n%3 == 0:
			radices[count] = 3
			n /= 3
		case n%2 == 0:
			radices[count] = 2
			n /= 2
		default:
			return 0
		}

		count++
		if count >= mixedRadixMaxStages {
			return 0
		}
	}

	return count
}

func mixedRadixRecursive[T Complex](dst, src []T, n, stride, step int, radices []int, twiddle, scratch []T, inverse bool) {
	if n == 1 {
		dst[0] = src[0]
		return
	}

	radix := radices[0]
	span := n / radix

	nextRadices := radices[1:]
	for j := range radix {
		mixedRadixRecursive(dst[j*span:], src[j*stride:], span, stride*radix, step*radix, nextRadices, twiddle, scratch[:span], inverse)
	}

	for k := range span {
		switch radix {
		case 2:
			w1 := twiddle[k*step]
			if inverse {
				w1 = conj(w1)
			}

			a0 := dst[k]
			a1 := w1 * dst[span+k]

			scratch[k] = a0 + a1
			scratch[span+k] = a0 - a1
		case 3:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
			}

			a0 := dst[k]
			a1 := w1 * dst[span+k]
			a2 := w2 * dst[2*span+k]

			var y0, y1, y2 T
			if inverse {
				y0, y1, y2 = butterfly3Inverse(a0, a1, a2)
			} else {
				y0, y1, y2 = butterfly3Forward(a0, a1, a2)
			}

			scratch[k] = y0
			scratch[span+k] = y1
			scratch[2*span+k] = y2
		case 4:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
			}

			a0 := dst[k]
			a1 := w1 * dst[span+k]
			a2 := w2 * dst[2*span+k]
			a3 := w3 * dst[3*span+k]

			var y0, y1, y2, y3 T
			if inverse {
				y0, y1, y2, y3 = butterfly4Inverse(a0, a1, a2, a3)
			} else {
				y0, y1, y2, y3 = butterfly4Forward(a0, a1, a2, a3)
			}

			scratch[k] = y0
			scratch[span+k] = y1
			scratch[2*span+k] = y2
			scratch[3*span+k] = y3
		case 5:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
			}

			a0 := dst[k]
			a1 := w1 * dst[span+k]
			a2 := w2 * dst[2*span+k]
			a3 := w3 * dst[3*span+k]
			a4 := w4 * dst[4*span+k]

			var y0, y1, y2, y3, y4 T
			if inverse {
				y0, y1, y2, y3, y4 = butterfly5Inverse(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = butterfly5Forward(a0, a1, a2, a3, a4)
			}

			scratch[k] = y0
			scratch[span+k] = y1
			scratch[2*span+k] = y2
			scratch[3*span+k] = y3
			scratch[4*span+k] = y4
		default:
			return
		}
	}

	copy(dst, scratch[:n])
}
