package kernels

func forwardStockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return stockhamForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseStockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return stockhamInverseComplex64(dst, src, twiddle, scratch, bitrev)
}

func forwardStockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return stockhamForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseStockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return stockhamInverseComplex128(dst, src, twiddle, scratch, bitrev)
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

	in = in[:n]
	out = out[:n]
	twiddle = twiddle[:n]

	stages := log2(n)
	halfN := n >> 1

	for s := range stages {
		m := 1 << (stages - s)
		half := m >> 1

		step := n / m

		kLimit := n / m
		for k := range kLimit {
			base := k * m

			outBase := k * half
			inBlock := in[base : base+m]
			outLo := out[outBase : outBase+half]

			outHi := out[outBase+halfN : outBase+halfN+half]
			for j := range half {
				a := inBlock[j]
				b := inBlock[j+half]
				tw := twiddle[j*step]
				outLo[j] = a + b
				outHi[j] = (a - b) * tw
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

		out = out[:n]
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

	in = in[:n]
	out = out[:n]
	twiddle = twiddle[:n]

	stages := log2(n)
	halfN := n >> 1

	for s := range stages {
		m := 1 << (stages - s)
		half := m >> 1

		step := n / m

		kLimit := n / m
		for k := range kLimit {
			base := k * m

			outBase := k * half
			inBlock := in[base : base+m]
			outLo := out[outBase : outBase+half]

			outHi := out[outBase+halfN : outBase+halfN+half]
			for j := range half {
				a := inBlock[j]
				b := inBlock[j+half]
				tw := conj(twiddle[j*step])
				outLo[j] = a + b
				outHi[j] = (a - b) * tw
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

		out = out[:n]
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

func stockhamInverseComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
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

	in = in[:n]
	out = out[:n]
	twiddle = twiddle[:n]

	stages := log2(n)
	halfN := n >> 1

	for s := range stages {
		m := 1 << (stages - s)
		half := m >> 1

		step := n / m

		kLimit := n / m
		for k := range kLimit {
			base := k * m

			outBase := k * half
			inBlock := in[base : base+m]
			outLo := out[outBase : outBase+half]

			outHi := out[outBase+halfN : outBase+halfN+half]
			for j := range half {
				a := inBlock[j]
				b := inBlock[j+half]
				tw := twiddle[j*step]
				tw = complex(real(tw), -imag(tw))
				outLo[j] = a + b
				outHi[j] = (a - b) * tw
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

		out = out[:n]
	}

	if !inIsDst {
		copy(dst, in)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}

func stockhamInverseComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
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

	in = in[:n]
	out = out[:n]
	twiddle = twiddle[:n]

	stages := log2(n)
	halfN := n >> 1

	for s := range stages {
		m := 1 << (stages - s)
		half := m >> 1

		step := n / m

		kLimit := n / m
		for k := range kLimit {
			base := k * m

			outBase := k * half
			inBlock := in[base : base+m]
			outLo := out[outBase : outBase+half]

			outHi := out[outBase+halfN : outBase+halfN+half]
			for j := range half {
				a := inBlock[j]
				b := inBlock[j+half]
				tw := twiddle[j*step]
				tw = complex(real(tw), -imag(tw))
				outLo[j] = a + b
				outHi[j] = (a - b) * tw
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

		out = out[:n]
	}

	if !inIsDst {
		copy(dst, in)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}

func sameSlice[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}

	return &a[0] == &b[0]
}


// Public exports for internal/fft re-export
var (
	ForwardStockhamComplex64  = forwardStockhamComplex64
	InverseStockhamComplex64  = inverseStockhamComplex64
	ForwardStockhamComplex128 = forwardStockhamComplex128
	InverseStockhamComplex128 = inverseStockhamComplex128
)

// SameSlice is exported for use by internal/fft
func SameSlice[T any](a, b []T) bool {
	return sameSlice(a, b)
}

// StockhamForward wraps the generic stockhamForward
func StockhamForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return stockhamForward(dst, src, twiddle, scratch, bitrev)
}

// StockhamInverse wraps stockhamInverseComplex64/128
func StockhamInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	var zero T
	switch any(zero).(type) {
	case complex64:
		return stockhamInverseComplex64(any(dst).([]complex64), any(src).([]complex64), any(twiddle).([]complex64), any(scratch).([]complex64), bitrev)
	case complex128:
		return stockhamInverseComplex128(any(dst).([]complex128), any(src).([]complex128), any(twiddle).([]complex128), any(scratch).([]complex128), bitrev)
	default:
		return false
	}
}
