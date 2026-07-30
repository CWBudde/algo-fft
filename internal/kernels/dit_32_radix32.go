package kernels

import (
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// forwardDIT32Radix32Complex64 computes a 32-point forward FFT using a single
// radix-32 pass (direct DFT) for complex64 data. This kernel performs the full
// transform without bit-reversal and returns false if any slice is too small.
func forwardDIT32Radix32Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 32

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for k := range n {
		var sum complex64

		for j := range n {
			idx := (j * k) & (n - 1)
			sum += mathpkg.MulComplex64(s[j], twiddle[idx])
		}

		work[k] = sum
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT32Radix32Complex64 computes a 32-point inverse FFT using a single
// radix-32 pass (direct DFT) for complex64 data. Uses conjugated twiddles and
// applies 1/N scaling. Returns false if any slice is too small.
func inverseDIT32Radix32Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 32

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for k := range n {
		var sum complex64

		for j := range n {
			idx := (j * k) & (n - 1)
			w := twiddle[idx]
			w = complex(real(w), -imag(w))
			sum += mathpkg.MulComplex64(s[j], w)
		}

		work[k] = sum
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	scale := float32(1.0 / float64(n))
	for i := range dst[:n] {
		dst[i] = complex(real(dst[i])*scale, imag(dst[i])*scale)
	}

	return true
}
