package kernels

import mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"

func forwardStockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	return stockhamForward[complex64](dst, src, twiddle, scratch)
}

func inverseStockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	return stockhamInverseComplex64(dst, src, twiddle, scratch)
}

func forwardStockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	return stockhamForward[complex128](dst, src, twiddle, scratch)
}

func inverseStockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	return stockhamInverseComplex128(dst, src, twiddle, scratch)
}

func stockhamForward[T Complex](dst, src, twiddle, scratch []T) bool {
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

	if !mathpkg.IsPowerOf2(n) {
		return false
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

func stockhamInverse[T Complex](dst, src, twiddle, scratch []T) bool {
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

	if !mathpkg.IsPowerOf2(n) {
		return false
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

func stockhamInverseComplex64(dst, src, twiddle, scratch []complex64) bool {
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

	if !mathpkg.IsPowerOf2(n) {
		return false
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

func stockhamInverseComplex128(dst, src, twiddle, scratch []complex128) bool {
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

	if !mathpkg.IsPowerOf2(n) {
		return false
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

// StockhamForward wraps the generic stockhamForward.
func StockhamForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return stockhamForward(dst, src, twiddle, scratch)
}

// StockhamInverse wraps stockhamInverseComplex64/128.
func StockhamInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	switch d := any(dst).(type) {
	case []complex64:
		s, _ := any(src).([]complex64)
		t, _ := any(twiddle).([]complex64)
		sc, _ := any(scratch).([]complex64)

		return stockhamInverseComplex64(d, s, t, sc)
	case []complex128:
		s, _ := any(src).([]complex128)
		t, _ := any(twiddle).([]complex128)
		sc, _ := any(scratch).([]complex128)

		return stockhamInverseComplex128(d, s, t, sc)
	default:
		return false
	}
}

// Precision-specific exports.
var (
	ForwardStockhamComplex64  = forwardStockhamComplex64
	InverseStockhamComplex64  = inverseStockhamComplex64
	ForwardStockhamComplex128 = forwardStockhamComplex128
	InverseStockhamComplex128 = inverseStockhamComplex128
)
