package algofft

// convolveRealT computes the linear convolution of a and b using real FFTs for
// the float type F and its matching complex type C. The dst slice must have
// length len(a)+len(b)-1.
//
// This is the one-shot entry point to the real convolution pipeline owned by
// RealConvolver: it validates, builds a throwaway RealConvolver, and runs it
// once.
func convolveRealT[F Float, C Complex](dst, a, b []F) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) == 0 || len(b) == 0 {
		return ErrInvalidLength
	}

	if len(dst) != len(a)+len(b)-1 {
		return ErrLengthMismatch
	}

	conv, err := NewRealConvolver[F, C](len(a), len(b))
	if err != nil {
		return err
	}

	return conv.Convolve(dst, a, b)
}

// ConvolveReal computes the linear convolution of a and b using real FFTs.
// The dst slice must have length len(a)+len(b)-1.
//
// Each call creates a fresh FFT plan; for repeated convolutions of
// same-length inputs, use a RealConvolver to reuse the plan and buffers.
func ConvolveReal(dst, a, b []float32) error {
	return convolveRealT[float32, complex64](dst, a, b)
}

// ConvolveReal64 computes the linear convolution of a and b using double-precision
// real FFTs. The dst slice must have length len(a)+len(b)-1.
func ConvolveReal64(dst, a, b []float64) error {
	return convolveRealT[float64, complex128](dst, a, b)
}
