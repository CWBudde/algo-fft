package kernels

// Radix-16 butterfly: a complete 16-point DFT, natural order in and out.
//
// It is factored as 4x4 rather than written out, because the flat form needs
// 120 complex multiplies and the factored form needs nine — and of those nine,
// four are free or half-price. Composing two verified radix-4 levels also means
// the only new arithmetic here is the twiddle layer between them, which is the
// part TestButterfly16MatchesReference actually has to catch.
//
// Writing n = 4*n1 + n0 and k = k0 + 4*k1 with all four digits in 0..3,
//
//	X[k0+4k1] = sum_n0 W16^(n0*k0) * W4^(n0*k1) * (sum_n1 x[n0+4n1] * W4^(n1*k0))
//
// so the inner sum is a radix-4 DFT down each stride-4 column, the W16^(n0*k0)
// factor is the twiddle layer, and the outer sum is a radix-4 DFT across the
// four columns. The outer DFT for a fixed k0 yields X[k0], X[k0+4], X[k0+8]
// and X[k0+12], which is why the stores below stride by four.
//
// The twiddle exponents n0*k0 for n0,k0 in 0..3 are 0,1,2,3,2,4,6,3,6,9. Only
// 1, 3 and 9 cost a general complex multiply, and W16^9 = -W16^1 because
// W16^8 = -1, so there are two distinct rotations rather than three. W16^4 = -i
// is a swap-and-negate, and W16^2 and W16^6 are (1-i)/sqrt2 and -(1+i)/sqrt2,
// which fold to two real multiplies each instead of four.

const (
	// cos(pi/8) and sin(pi/8): the W16^1 rotation, and W16^3 with the parts swapped.
	cos16 = 0.9238795325112867
	sin16 = 0.3826834323650898
	// sqrt(2)/2: the shared magnitude of W16^2 and W16^6.
	sqrt2Half16 = 0.7071067811865476
)

// mulW16Pow1Complex64 multiplies by W16^1 = cos(pi/8) - i*sin(pi/8).
//
// Written out in float32 rather than as mathpkg.MulComplex64 against a constant
// because the generic helper costs 100 inline units against a budget of 80, and
// the ladder calls this once per butterfly. The explicit form is the same four
// multiplies and two adds, it keeps the float64 widening away (which is the
// whole point of MulComplex64), and it inlines.
func mulW16Pow1Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex(x*cos16+y*sin16, y*cos16-x*sin16)
}

// mulW16Pow3Complex64 multiplies by W16^3 = sin(pi/8) - i*cos(pi/8).
func mulW16Pow3Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex(x*sin16+y*cos16, y*sin16-x*cos16)
}

// mulW16Pow2Complex64 multiplies by W16^2 = (1-i)/sqrt(2). Two real multiplies:
// z*(1-i) = (x+y) + i*(y-x), then scale.
func mulW16Pow2Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex((x+y)*sqrt2Half16, (y-x)*sqrt2Half16)
}

// mulW16Pow6Complex64 multiplies by W16^6 = -(1+i)/sqrt(2). Two real multiplies:
// z*(1+i) = (x-y) + i*(x+y), then scale and negate.
func mulW16Pow6Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex((y-x)*sqrt2Half16, -(x+y)*sqrt2Half16)
}

// mulNegIComplex64 multiplies by W16^4 = -i, which costs no multiply at all.
func mulNegIComplex64(z complex64) complex64 {
	return complex(imag(z), -real(z))
}

// mulPosIComplex64 multiplies by +i, the inverse-direction twin of the above.
func mulPosIComplex64(z complex64) complex64 {
	return complex(-imag(z), real(z))
}

// butterfly16ForwardComplex64 replaces x with its 16-point DFT, natural order
// in and out.
func butterfly16ForwardComplex64(x *[16]complex64) {
	// Level 1: a radix-4 DFT down each stride-4 column. b<n0><k0>.
	b00, b01, b02, b03 := butterfly4ForwardComplex64(x[0], x[4], x[8], x[12])
	b10, b11, b12, b13 := butterfly4ForwardComplex64(x[1], x[5], x[9], x[13])
	b20, b21, b22, b23 := butterfly4ForwardComplex64(x[2], x[6], x[10], x[14])
	b30, b31, b32, b33 := butterfly4ForwardComplex64(x[3], x[7], x[11], x[15])

	// Twiddle layer: multiply b<n0><k0> by W16^(n0*k0). Row 0 and column 0 are
	// W16^0 = 1 and drop out entirely.
	c11 := mulW16Pow1Complex64(b11)
	c12 := mulW16Pow2Complex64(b12)
	c13 := mulW16Pow3Complex64(b13)

	c21 := mulW16Pow2Complex64(b21)
	c22 := mulNegIComplex64(b22)
	c23 := mulW16Pow6Complex64(b23)

	c31 := mulW16Pow3Complex64(b31)
	c32 := mulW16Pow6Complex64(b32)
	c33 := -mulW16Pow1Complex64(b33) // W16^9 = -W16^1

	// Level 2: a radix-4 DFT across the four columns, for each k0. Output k1 of
	// the k0-th call is X[k0+4*k1].
	x[0], x[4], x[8], x[12] = butterfly4ForwardComplex64(b00, b10, b20, b30)
	x[1], x[5], x[9], x[13] = butterfly4ForwardComplex64(b01, c11, c21, c31)
	x[2], x[6], x[10], x[14] = butterfly4ForwardComplex64(b02, c12, c22, c32)
	x[3], x[7], x[11], x[15] = butterfly4ForwardComplex64(b03, c13, c23, c33)
}

// butterfly16InverseComplex64 replaces x with its unnormalised 16-point inverse
// DFT. Every twiddle is the conjugate of the forward one; the 1/n scaling is
// the ladder's business, not the butterfly's.
func butterfly16InverseComplex64(x *[16]complex64) {
	b00, b01, b02, b03 := butterfly4InverseComplex64(x[0], x[4], x[8], x[12])
	b10, b11, b12, b13 := butterfly4InverseComplex64(x[1], x[5], x[9], x[13])
	b20, b21, b22, b23 := butterfly4InverseComplex64(x[2], x[6], x[10], x[14])
	b30, b31, b32, b33 := butterfly4InverseComplex64(x[3], x[7], x[11], x[15])

	c11 := mulW16PowNeg1Complex64(b11)
	c12 := mulW16PowNeg2Complex64(b12)
	c13 := mulW16PowNeg3Complex64(b13)

	c21 := mulW16PowNeg2Complex64(b21)
	c22 := mulPosIComplex64(b22)
	c23 := mulW16PowNeg6Complex64(b23)

	c31 := mulW16PowNeg3Complex64(b31)
	c32 := mulW16PowNeg6Complex64(b32)
	c33 := -mulW16PowNeg1Complex64(b33)

	x[0], x[4], x[8], x[12] = butterfly4InverseComplex64(b00, b10, b20, b30)
	x[1], x[5], x[9], x[13] = butterfly4InverseComplex64(b01, c11, c21, c31)
	x[2], x[6], x[10], x[14] = butterfly4InverseComplex64(b02, c12, c22, c32)
	x[3], x[7], x[11], x[15] = butterfly4InverseComplex64(b03, c13, c23, c33)
}

// mulW16PowNeg1Complex64 multiplies by W16^-1 = cos(pi/8) + i*sin(pi/8).
func mulW16PowNeg1Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex(x*cos16-y*sin16, y*cos16+x*sin16)
}

// mulW16PowNeg3Complex64 multiplies by W16^-3 = sin(pi/8) + i*cos(pi/8).
func mulW16PowNeg3Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex(x*sin16-y*cos16, y*sin16+x*cos16)
}

// mulW16PowNeg2Complex64 multiplies by W16^-2 = (1+i)/sqrt(2).
func mulW16PowNeg2Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex((x-y)*sqrt2Half16, (x+y)*sqrt2Half16)
}

// mulW16PowNeg6Complex64 multiplies by W16^-6 = -(1-i)/sqrt(2).
func mulW16PowNeg6Complex64(z complex64) complex64 {
	x, y := real(z), imag(z)

	return complex(-(x+y)*sqrt2Half16, (x-y)*sqrt2Half16)
}
