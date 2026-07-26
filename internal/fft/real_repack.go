package fft

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

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

	start := max(inverseRepackComplex64SIMD(dst, src, weight), 1)

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

	start := max(inverseRepackComplex128SIMD(dst, src, weight), 1)

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
		// Doubling component-wise: `2*u` is a complex64 multiply, so it would
		// widen to complex128 and round back for the same two products (see
		// math.MulComplex64).
		det := complex64(1) - complex(2*real(u), 2*imag(u))
		// det is on the unit circle, so 1/det == conj(det)
		invDet := complex(real(det), -imag(det))

		// mathpkg.MulComplex64 rather than `*` for the same reason: the
		// operator promotes every one of these six products to double
		// precision. Multiplication is commutative in IEEE arithmetic, so
		// naming the four cross products costs nothing and computes each once.
		xkOne := mathpkg.MulComplex64(xk, oneMinusU)
		xkU := mathpkg.MulComplex64(xk, u)
		xmkcOne := mathpkg.MulComplex64(xmkc, oneMinusU)
		xmkcU := mathpkg.MulComplex64(xmkc, u)

		a := mathpkg.MulComplex64(xkOne-xmkcU, invDet)
		b := mathpkg.MulComplex64(xmkcOne-xkU, invDet)

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
		// Doubled component-wise, as in the complex64 twin: two products
		// instead of the operator's four.
		det := complex128(1) - complex(2*real(u), 2*imag(u))
		// det is on the unit circle, so 1/det == conj(det)
		invDet := complex(real(det), -imag(det))

		// MulComplex128 is the plain operator; naming the four cross products
		// keeps this loop line-for-line comparable with its complex64 twin.
		xkOne := mathpkg.MulComplex128(xk, oneMinusU)
		xkU := mathpkg.MulComplex128(xk, u)
		xmkcOne := mathpkg.MulComplex128(xmkc, oneMinusU)
		xmkcU := mathpkg.MulComplex128(xmkc, u)

		a := mathpkg.MulComplex128(xkOne-xmkcU, invDet)
		b := mathpkg.MulComplex128(xmkcOne-xkU, invDet)

		dst[k] = a
		if k != m {
			dst[m] = complex(real(b), -imag(b))
		}
	}
}
