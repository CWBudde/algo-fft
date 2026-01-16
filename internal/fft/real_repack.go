package fft

// RepackInverseComplex64 reconstructs the packed buffer for an inverse real FFT.
// dst has length n/2, src and weight have length n/2+1.
func RepackInverseComplex64(dst, src, weight []complex64) {
	if len(dst) == 0 {
		return
	}

	x0 := real(src[0])
	xh := real(src[len(dst)])
	dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

	if len(dst) < 2 {
		return
	}

	start := inverseRepackComplex64SIMD(dst, src, weight)
	if start < 1 {
		start = 1
	}

	inverseRepackComplex64Generic(dst, src, weight, start)
}

// RepackInverseComplex128 reconstructs the packed buffer for an inverse real FFT.
// dst has length n/2, src and weight have length n/2+1.
func RepackInverseComplex128(dst, src, weight []complex128) {
	if len(dst) == 0 {
		return
	}

	x0 := real(src[0])
	xh := real(src[len(dst)])
	dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

	if len(dst) < 2 {
		return
	}

	start := inverseRepackComplex128SIMD(dst, src, weight)
	if start < 1 {
		start = 1
	}

	inverseRepackComplex128Generic(dst, src, weight, start)
}

func inverseRepackComplex64Generic(dst, src, weight []complex64, start int) {
	half := len(dst)
	for k := start; k < half; k++ {
		m := half - k
		if k > m {
			continue
		}

		xk := src[k]
		xmk := src[m]
		xmkc := complex(real(xmk), -imag(xmk))

		u := weight[k]
		oneMinusU := complex64(1) - u
		det := complex64(1) - 2*u
		// det is on the unit circle, so 1/det == conj(det)
		invDet := complex(real(det), -imag(det))

		a := (xk*oneMinusU - xmkc*u) * invDet
		b := (oneMinusU*xmkc - u*xk) * invDet

		dst[k] = a
		if k != m {
			dst[m] = complex(real(b), -imag(b))
		}
	}
}

func inverseRepackComplex128Generic(dst, src, weight []complex128, start int) {
	half := len(dst)
	for k := start; k < half; k++ {
		m := half - k
		if k > m {
			continue
		}

		xk := src[k]
		xmk := src[m]
		xmkc := complex(real(xmk), -imag(xmk))

		u := weight[k]
		oneMinusU := complex128(1) - u
		det := complex128(1) - 2*u
		// det is on the unit circle, so 1/det == conj(det)
		invDet := complex(real(det), -imag(det))

		a := (xk*oneMinusU - xmkc*u) * invDet
		b := (oneMinusU*xmkc - u*xk) * invDet

		dst[k] = a
		if k != m {
			dst[m] = complex(real(b), -imag(b))
		}
	}
}
