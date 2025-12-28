package fft

func forwardRadix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix4Forward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix4Inverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardRadix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix4Forward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix4Inverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func radix4Forward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix4Transform(dst, src, twiddle, scratch, bitrev, false)
}

func radix4Inverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix4Transform(dst, src, twiddle, scratch, bitrev, true)
}

func radix4Transform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
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

	if !isPowerOf4(n) {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	digits := log2(n) / 2
	for i := range n {
		work[i] = src[reverseBase4(i, digits)]
	}

	for size := 4; size <= n; size *= 4 {
		quarter := size / 4

		step := n / size
		for base := 0; base < n; base += size {
			for j := range quarter {
				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]
				w3 := twiddle[3*j*step]

				if inverse {
					w1 = conj(w1)
					w2 = conj(w2)
					w3 = conj(w3)
				}

				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]
				a3 := w3 * work[idx3]

				var y0, y1, y2, y3 T
				if inverse {
					y0, y1, y2, y3 = butterfly4Inverse(a0, a1, a2, a3)
				} else {
					y0, y1, y2, y3 = butterfly4Forward(a0, a1, a2, a3)
				}

				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
				work[idx3] = y3
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

func butterfly4Forward[T Complex](a0, a1, a2, a3 T) (T, T, T, T) {
	t0 := a0 + a2
	t1 := a0 - a2
	t2 := a1 + a3
	t3 := a1 - a3

	y0 := t0 + t2
	y2 := t0 - t2
	y1 := t1 + mulNegI(t3)
	y3 := t1 + mulI(t3)

	return y0, y1, y2, y3
}

func butterfly4Inverse[T Complex](a0, a1, a2, a3 T) (T, T, T, T) {
	t0 := a0 + a2
	t1 := a0 - a2
	t2 := a1 + a3
	t3 := a1 - a3

	y0 := t0 + t2
	y2 := t0 - t2
	y1 := t1 + mulI(t3)
	y3 := t1 + mulNegI(t3)

	return y0, y1, y2, y3
}

func mulI[T Complex](value T) T {
	switch v := any(value).(type) {
	case complex64:
		return any(complex(-imag(v), real(v))).(T)
	case complex128:
		return any(complex(-imag(v), real(v))).(T)
	default:
		panic("unsupported complex type")
	}
}

func mulNegI[T Complex](value T) T {
	switch v := any(value).(type) {
	case complex64:
		return any(complex(imag(v), -real(v))).(T)
	case complex128:
		return any(complex(imag(v), -real(v))).(T)
	default:
		panic("unsupported complex type")
	}
}

func reverseBase4(x, digits int) int {
	result := 0
	for range digits {
		result = (result << 2) | (x & 0x3)
		x >>= 2
	}

	return result
}
