package algofft

// Complex is a type constraint for the complex number types supported by the
// FFT. It is declared in this package (not aliased from an internal package)
// so internal refactors cannot change the public API.
type Complex interface {
	complex64 | complex128
}

// Float is a type constraint for the floating-point types used in real FFT
// operations. Float and Complex parameters are always paired:
// float32 with complex64, float64 with complex128.
type Float interface {
	float32 | float64
}
