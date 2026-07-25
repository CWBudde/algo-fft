package math

// Go's compiler does not implement scalar `complex64 * complex64` in single
// precision. It widens all four float32 components to float64, multiplies in
// double precision, and rounds the two results back:
//
//	CVTSS2SD ×4, MULSD ×3, VFMADD231SD, SUBSD, CVTSD2SS ×2   (12 instructions)
//
// against six for the same expression on complex128. Addition, subtraction and
// conjugation are unaffected — only the multiply promotes.
//
// The consequence is that any FFT stage written as scalar Go costs *more* in
// complex64 than in complex128. That is what the external sweep in PLAN.md
// P5.0 measured: complex64 lost to complex128 at 20 of 23 non-power-of-two
// lengths, because those lengths run the mixed-radix, Bluestein and Rader
// paths — whose odd-radix stages and pointwise products are scalar Go — while
// power-of-two lengths run inside hand-written float32 SIMD codelets, where
// complex64 is genuinely half the width.
//
// MulComplex64 multiplies the components directly, which keeps the operation in
// single precision (MULSS ×3, VFMADD231SS, SUBSS) and matches the arithmetic
// the SIMD codelets already perform, so the Go and assembly paths round the
// same way instead of differing by the double-rounded product.
//
// Use it for scalar complex64 multiplication anywhere in a transform's inner
// loop. Slice-wide products should prefer the SIMD entry points in
// internal/fft (ComplexMulArrayComplex64 and friends).
func MulComplex64(a, b complex64) complex64 {
	ar, ai := real(a), imag(a)
	br, bi := real(b), imag(b)

	return complex(ar*br-ai*bi, ar*bi+ai*br)
}

// MulComplex128 multiplies two complex128 values. Double-precision complex
// multiplication is already native, so this is the plain operator. It exists so
// that complex64 code written against MulComplex64 has a symmetric twin: the
// genkernels rewrite (Complex64 -> Complex128) maps call sites onto it
// automatically, and hand-written pairs stay line-for-line comparable.
func MulComplex128(a, b complex128) complex128 {
	return a * b
}
