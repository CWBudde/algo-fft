package kernels

// Radix-8 butterfly for the mixed-radix engine.
//
// Given eight already-twiddled inputs a0..a7 (a_j = W_N^{jk}·x_j), the
// butterfly computes the 8-point DFT y_m = Σ_j a_j·W_8^{jm} using the same
// even/odd decomposition as the size-512 radix-8 codelet
// (dit_512_radix8.go): 2×radix-4 halves combined with the three internal
// eighth-roots. All internal rotations (±i, W_8^1, W_8^3) are hardcoded
// constants, so the butterfly costs 4 real multiplies per W_8^{1,3} rotation
// instead of full complex multiplies.

// root2Over2 is √2/2, the real (and ± imaginary) part of the odd
// eighth-roots of unity W_8^1 and W_8^3.
const root2Over2 = 0.70710678118654752440084436210485

func butterfly8ForwardComplex64(x0, x1, x2, x3, x4, x5, x6, x7 complex64) (complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64) {
	a0 := x0 + x4
	a1 := x0 - x4
	a2 := x2 + x6
	a3 := x2 - x6
	a4 := x1 + x5
	a5 := x1 - x5
	a6 := x3 + x7
	a7 := x3 - x7

	e0 := a0 + a2
	e2 := a0 - a2
	e1 := a1 + complex(imag(a3), -real(a3)) // a1 − i·a3
	e3 := a1 + complex(-imag(a3), real(a3)) // a1 + i·a3

	o0 := a4 + a6
	o2 := a4 - a6
	o1 := a5 + complex(imag(a7), -real(a7)) // a5 − i·a7
	o3 := a5 + complex(-imag(a7), real(a7)) // a5 + i·a7

	// t1 = W_8^1·o1, t2 = W_8^2·o2 = −i·o2, t3 = W_8^3·o3
	t1 := complex(root2Over2*(real(o1)+imag(o1)), root2Over2*(imag(o1)-real(o1)))
	t2 := complex(imag(o2), -real(o2))
	t3 := complex(root2Over2*(imag(o3)-real(o3)), -root2Over2*(real(o3)+imag(o3)))

	return e0 + o0, e1 + t1, e2 + t2, e3 + t3, e0 - o0, e1 - t1, e2 - t2, e3 - t3
}

func butterfly8InverseComplex64(x0, x1, x2, x3, x4, x5, x6, x7 complex64) (complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64) {
	a0 := x0 + x4
	a1 := x0 - x4
	a2 := x2 + x6
	a3 := x2 - x6
	a4 := x1 + x5
	a5 := x1 - x5
	a6 := x3 + x7
	a7 := x3 - x7

	e0 := a0 + a2
	e2 := a0 - a2
	e1 := a1 + complex(-imag(a3), real(a3)) // a1 + i·a3
	e3 := a1 + complex(imag(a3), -real(a3)) // a1 − i·a3

	o0 := a4 + a6
	o2 := a4 - a6
	o1 := a5 + complex(-imag(a7), real(a7)) // a5 + i·a7
	o3 := a5 + complex(imag(a7), -real(a7)) // a5 − i·a7

	// Conjugated roots: t1 = conj(W_8^1)·o1, t2 = i·o2, t3 = conj(W_8^3)·o3
	t1 := complex(root2Over2*(real(o1)-imag(o1)), root2Over2*(imag(o1)+real(o1)))
	t2 := complex(-imag(o2), real(o2))
	t3 := complex(-root2Over2*(real(o3)+imag(o3)), root2Over2*(real(o3)-imag(o3)))

	return e0 + o0, e1 + t1, e2 + t2, e3 + t3, e0 - o0, e1 - t1, e2 - t2, e3 - t3
}

// Public exports for internal/fft - type-specific functions for direct calls.

// Butterfly8ForwardComplex64 computes the forward 8-point DFT of eight
// already-twiddled complex64 inputs.
func Butterfly8ForwardComplex64(x0, x1, x2, x3, x4, x5, x6, x7 complex64) (complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64) {
	return butterfly8ForwardComplex64(x0, x1, x2, x3, x4, x5, x6, x7)
}

// Butterfly8InverseComplex64 computes the inverse (unscaled) 8-point DFT of
// eight already-twiddled complex64 inputs.
func Butterfly8InverseComplex64(x0, x1, x2, x3, x4, x5, x6, x7 complex64) (complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64) {
	return butterfly8InverseComplex64(x0, x1, x2, x3, x4, x5, x6, x7)
}
