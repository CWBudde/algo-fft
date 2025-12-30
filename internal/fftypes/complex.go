package fftypes

// Complex is a type constraint for complex number types supported by the FFT.
type Complex interface {
	complex64 | complex128
}

// Float is a type constraint for floating-point types used in real FFT operations.
type Float interface {
	float32 | float64
}
