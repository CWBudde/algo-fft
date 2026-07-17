package kernels

import (
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// Split-radix (2/4) decimation-in-time FFT for power-of-two sizes.
//
// A size-n transform splits into one size-n/2 transform over the even
// samples and two size-n/4 transforms over the odd samples U[m] = x[4m+1]
// and Z[m] = x[4m+3]. With W = exp(-2πi/n), for k in [0, n/4):
//
//	a  = W^k  · U[k],  b = W^3k · Z[k]
//	X[k]       = E[k]     + (a + b)
//	X[k+n/2]   = E[k]     - (a + b)
//	X[k+n/4]   = E[k+n/4] - i(a - b)
//	X[k+3n/4]  = E[k+n/4] + i(a - b)
//
// (signs of the ±i terms flip for the inverse). This is the classical
// lowest-operation-count decomposition for power-of-two sizes: ~33% fewer
// real operations than radix-2 and ~10% fewer than radix-4. The recursion
// writes E into dst[0:n/2], U into dst[n/2:3n/4], and Z into dst[3n/4:n],
// so the combine step above is in-place and leaves dst in natural order —
// no bit-reversal pass is needed.

// splitRadixMinRecurse is the size at which the recursion bottoms out into
// unrolled leaves (sizes 1, 2, 4).
const splitRadixMinRecurse = 8

func splitRadixForwardRecurseComplex64(dst, src []complex64, stride, step int, twiddle []complex64) {
	n := len(dst)

	if n < splitRadixMinRecurse {
		switch n {
		case 1:
			dst[0] = src[0]
		case 2:
			a, b := src[0], src[stride]
			dst[0] = a + b
			dst[1] = a - b
		case 4:
			e0 := src[0] + src[2*stride]
			e1 := src[0] - src[2*stride]
			u, z := src[stride], src[3*stride]
			t1 := u + z
			d := u - z
			t2 := complex(imag(d), -real(d)) // -i·d
			dst[0] = e0 + t1
			dst[2] = e0 - t1
			dst[1] = e1 + t2
			dst[3] = e1 - t2
		}

		return
	}

	half, quarter := n/2, n/4

	splitRadixForwardRecurseComplex64(dst[:half], src, 2*stride, 2*step, twiddle)
	splitRadixForwardRecurseComplex64(dst[half:half+quarter], src[stride:], 4*stride, 4*step, twiddle)
	splitRadixForwardRecurseComplex64(dst[half+quarter:n], src[3*stride:], 4*stride, 4*step, twiddle)

	// k = 0: both twiddles are 1.
	{
		a, b := dst[half], dst[half+quarter]
		t1 := a + b
		d := a - b
		t2 := complex(imag(d), -real(d))
		e0, e1 := dst[0], dst[quarter]
		dst[0] = e0 + t1
		dst[half] = e0 - t1
		dst[quarter] = e1 + t2
		dst[half+quarter] = e1 - t2
	}

	for k := 1; k < quarter; k++ {
		w1 := twiddle[k*step]
		w3 := twiddle[3*k*step]
		a := w1 * dst[half+k]
		b := w3 * dst[half+quarter+k]
		t1 := a + b
		d := a - b
		t2 := complex(imag(d), -real(d))
		e0, e1 := dst[k], dst[quarter+k]
		dst[k] = e0 + t1
		dst[half+k] = e0 - t1
		dst[quarter+k] = e1 + t2
		dst[half+quarter+k] = e1 - t2
	}
}

func splitRadixInverseRecurseComplex64(dst, src []complex64, stride, step int, twiddle []complex64) {
	n := len(dst)

	if n < splitRadixMinRecurse {
		switch n {
		case 1:
			dst[0] = src[0]
		case 2:
			a, b := src[0], src[stride]
			dst[0] = a + b
			dst[1] = a - b
		case 4:
			e0 := src[0] + src[2*stride]
			e1 := src[0] - src[2*stride]
			u, z := src[stride], src[3*stride]
			t1 := u + z
			d := u - z
			t2 := complex(-imag(d), real(d)) // +i·d
			dst[0] = e0 + t1
			dst[2] = e0 - t1
			dst[1] = e1 + t2
			dst[3] = e1 - t2
		}

		return
	}

	half, quarter := n/2, n/4

	splitRadixInverseRecurseComplex64(dst[:half], src, 2*stride, 2*step, twiddle)
	splitRadixInverseRecurseComplex64(dst[half:half+quarter], src[stride:], 4*stride, 4*step, twiddle)
	splitRadixInverseRecurseComplex64(dst[half+quarter:n], src[3*stride:], 4*stride, 4*step, twiddle)

	{
		a, b := dst[half], dst[half+quarter]
		t1 := a + b
		d := a - b
		t2 := complex(-imag(d), real(d))
		e0, e1 := dst[0], dst[quarter]
		dst[0] = e0 + t1
		dst[half] = e0 - t1
		dst[quarter] = e1 + t2
		dst[half+quarter] = e1 - t2
	}

	for k := 1; k < quarter; k++ {
		w1 := mathpkg.Conj(twiddle[k*step])
		w3 := mathpkg.Conj(twiddle[3*k*step])
		a := w1 * dst[half+k]
		b := w3 * dst[half+quarter+k]
		t1 := a + b
		d := a - b
		t2 := complex(-imag(d), real(d))
		e0, e1 := dst[k], dst[quarter+k]
		dst[k] = e0 + t1
		dst[half+k] = e0 - t1
		dst[quarter+k] = e1 + t2
		dst[half+quarter+k] = e1 - t2
	}
}

func splitRadixForwardRecurseComplex128(dst, src []complex128, stride, step int, twiddle []complex128) {
	n := len(dst)

	if n < splitRadixMinRecurse {
		switch n {
		case 1:
			dst[0] = src[0]
		case 2:
			a, b := src[0], src[stride]
			dst[0] = a + b
			dst[1] = a - b
		case 4:
			e0 := src[0] + src[2*stride]
			e1 := src[0] - src[2*stride]
			u, z := src[stride], src[3*stride]
			t1 := u + z
			d := u - z
			t2 := complex(imag(d), -real(d)) // -i·d
			dst[0] = e0 + t1
			dst[2] = e0 - t1
			dst[1] = e1 + t2
			dst[3] = e1 - t2
		}

		return
	}

	half, quarter := n/2, n/4

	splitRadixForwardRecurseComplex128(dst[:half], src, 2*stride, 2*step, twiddle)
	splitRadixForwardRecurseComplex128(dst[half:half+quarter], src[stride:], 4*stride, 4*step, twiddle)
	splitRadixForwardRecurseComplex128(dst[half+quarter:n], src[3*stride:], 4*stride, 4*step, twiddle)

	{
		a, b := dst[half], dst[half+quarter]
		t1 := a + b
		d := a - b
		t2 := complex(imag(d), -real(d))
		e0, e1 := dst[0], dst[quarter]
		dst[0] = e0 + t1
		dst[half] = e0 - t1
		dst[quarter] = e1 + t2
		dst[half+quarter] = e1 - t2
	}

	for k := 1; k < quarter; k++ {
		w1 := twiddle[k*step]
		w3 := twiddle[3*k*step]
		a := w1 * dst[half+k]
		b := w3 * dst[half+quarter+k]
		t1 := a + b
		d := a - b
		t2 := complex(imag(d), -real(d))
		e0, e1 := dst[k], dst[quarter+k]
		dst[k] = e0 + t1
		dst[half+k] = e0 - t1
		dst[quarter+k] = e1 + t2
		dst[half+quarter+k] = e1 - t2
	}
}

func splitRadixInverseRecurseComplex128(dst, src []complex128, stride, step int, twiddle []complex128) {
	n := len(dst)

	if n < splitRadixMinRecurse {
		switch n {
		case 1:
			dst[0] = src[0]
		case 2:
			a, b := src[0], src[stride]
			dst[0] = a + b
			dst[1] = a - b
		case 4:
			e0 := src[0] + src[2*stride]
			e1 := src[0] - src[2*stride]
			u, z := src[stride], src[3*stride]
			t1 := u + z
			d := u - z
			t2 := complex(-imag(d), real(d)) // +i·d
			dst[0] = e0 + t1
			dst[2] = e0 - t1
			dst[1] = e1 + t2
			dst[3] = e1 - t2
		}

		return
	}

	half, quarter := n/2, n/4

	splitRadixInverseRecurseComplex128(dst[:half], src, 2*stride, 2*step, twiddle)
	splitRadixInverseRecurseComplex128(dst[half:half+quarter], src[stride:], 4*stride, 4*step, twiddle)
	splitRadixInverseRecurseComplex128(dst[half+quarter:n], src[3*stride:], 4*stride, 4*step, twiddle)

	{
		a, b := dst[half], dst[half+quarter]
		t1 := a + b
		d := a - b
		t2 := complex(-imag(d), real(d))
		e0, e1 := dst[0], dst[quarter]
		dst[0] = e0 + t1
		dst[half] = e0 - t1
		dst[quarter] = e1 + t2
		dst[half+quarter] = e1 - t2
	}

	for k := 1; k < quarter; k++ {
		w1 := mathpkg.Conj(twiddle[k*step])
		w3 := mathpkg.Conj(twiddle[3*k*step])
		a := w1 * dst[half+k]
		b := w3 * dst[half+quarter+k]
		t1 := a + b
		d := a - b
		t2 := complex(-imag(d), real(d))
		e0, e1 := dst[k], dst[quarter+k]
		dst[k] = e0 + t1
		dst[half+k] = e0 - t1
		dst[quarter+k] = e1 + t2
		dst[half+quarter+k] = e1 - t2
	}
}

// splitRadixValid reports whether the slice lengths admit a size-n
// split-radix transform (power-of-two n with full-size twiddle table).
func splitRadixValid[T Complex](dst, src, twiddle, scratch []T) (int, bool) {
	n := len(src)
	if n == 0 {
		return 0, true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || !mathpkg.IsPowerOf2(n) {
		return 0, false
	}

	return n, true
}

// ForwardSplitRadixComplex64 computes an unscaled forward FFT via the
// split-radix recursion. dst and src may alias (scratch is used as the work
// buffer in that case). Output is in natural order.
func ForwardSplitRadixComplex64(dst, src, twiddle, scratch []complex64) bool {
	n, ok := splitRadixValid(dst, src, twiddle, scratch)
	if !ok || n == 0 {
		return ok
	}

	work := dst[:n]
	if SameSlice(dst, src) {
		work = scratch[:n]
	}

	splitRadixForwardRecurseComplex64(work, src[:n], 1, len(twiddle)/n, twiddle)

	if SameSlice(dst, src) {
		copy(dst[:n], work)
	}

	return true
}

// InverseSplitRadixComplex64 is the inverse counterpart of
// ForwardSplitRadixComplex64; the output is scaled by 1/n per the library
// convention.
func InverseSplitRadixComplex64(dst, src, twiddle, scratch []complex64) bool {
	n, ok := splitRadixValid(dst, src, twiddle, scratch)
	if !ok || n == 0 {
		return ok
	}

	work := dst[:n]
	if SameSlice(dst, src) {
		work = scratch[:n]
	}

	splitRadixInverseRecurseComplex64(work, src[:n], 1, len(twiddle)/n, twiddle)

	scale := complex(float32(1)/float32(n), 0)
	for i, v := range work {
		dst[i] = v * scale
	}

	return true
}

// ForwardSplitRadixComplex128 is the complex128 variant of
// ForwardSplitRadixComplex64.
func ForwardSplitRadixComplex128(dst, src, twiddle, scratch []complex128) bool {
	n, ok := splitRadixValid(dst, src, twiddle, scratch)
	if !ok || n == 0 {
		return ok
	}

	work := dst[:n]
	if SameSlice(dst, src) {
		work = scratch[:n]
	}

	splitRadixForwardRecurseComplex128(work, src[:n], 1, len(twiddle)/n, twiddle)

	if SameSlice(dst, src) {
		copy(dst[:n], work)
	}

	return true
}

// InverseSplitRadixComplex128 is the complex128 variant of
// InverseSplitRadixComplex64.
func InverseSplitRadixComplex128(dst, src, twiddle, scratch []complex128) bool {
	n, ok := splitRadixValid(dst, src, twiddle, scratch)
	if !ok || n == 0 {
		return ok
	}

	work := dst[:n]
	if SameSlice(dst, src) {
		work = scratch[:n]
	}

	splitRadixInverseRecurseComplex128(work, src[:n], 1, len(twiddle)/n, twiddle)

	scale := complex(1/float64(n), 0)
	for i, v := range work {
		dst[i] = v * scale
	}

	return true
}
