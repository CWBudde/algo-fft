package fft

func forwardStockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return stockhamForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseStockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return stockhamInverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardStockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return stockhamForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseStockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return stockhamInverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func stockhamForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
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

	in := src
	out := dst
	same := sameSlice(dst, src)
	inIsDst := same
	outIsDst := true

	if same {
		out = scratch
		outIsDst = false
	}

	stages := log2(n)
	for s := range stages {
		m := 1 << (stages - s)
		half := m >> 1

		step := n / m
		for k := range n / m {
			base := k * m

			outBase := k * half
			for j := range half {
				a := in[base+j]
				b := in[base+j+half]
				tw := twiddle[j*step]
				out[outBase+j] = a + b
				out[outBase+j+n/2] = (a - b) * tw
			}
		}

		in = out

		inIsDst = outIsDst
		if outIsDst {
			out = scratch
			outIsDst = false
		} else {
			out = dst
			outIsDst = true
		}
	}

	if !inIsDst {
		copy(dst, in)
	}

	return true
}

func stockhamInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
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

	in := src
	out := dst
	same := sameSlice(dst, src)
	inIsDst := same
	outIsDst := true

	if same {
		out = scratch
		outIsDst = false
	}

	stages := log2(n)
	for s := range stages {
		m := 1 << (stages - s)
		half := m >> 1

		step := n / m
		for k := range n / m {
			base := k * m

			outBase := k * half
			for j := range half {
				a := in[base+j]
				b := in[base+j+half]
				tw := conj(twiddle[j*step])
				out[outBase+j] = a + b
				out[outBase+j+n/2] = (a - b) * tw
			}
		}

		in = out

		inIsDst = outIsDst
		if outIsDst {
			out = scratch
			outIsDst = false
		} else {
			out = dst
			outIsDst = true
		}
	}

	if !inIsDst {
		copy(dst, in)
	}

	scale := complexFromFloat64[T](1.0/float64(n), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}

func conj[T Complex](value T) T {
	switch v := any(value).(type) {
	case complex64:
		return any(complex(real(v), -imag(v))).(T)
	case complex128:
		return any(complex(real(v), -imag(v))).(T)
	default:
		panic("unsupported complex type")
	}
}

func sameSlice[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}

	return &a[0] == &b[0]
}
