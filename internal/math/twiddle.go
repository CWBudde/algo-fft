package math

import "math"

// ComputeTwiddleFactors returns the precomputed twiddle factors (roots of unity)
// for a size-n FFT: W_n^k = exp(-2πik/n) for k = 0..n-1.
func ComputeTwiddleFactors[T Complex](n int) []T {
	if n <= 0 {
		return nil
	}

	twiddle := make([]T, n)
	for k := range n {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		re := math.Cos(angle)
		im := math.Sin(angle)
		twiddle[k] = complexFromFloat64[T](re, im)
	}

	return twiddle
}

// ComplexFromFloat64 creates a complex number of type T from float64 components.
func ComplexFromFloat64[T Complex](re, im float64) T {
	var zero T

	switch any(zero).(type) {
	case complex64:
		result, _ := any(complex(float32(re), float32(im))).(T)
		return result
	case complex128:
		result, _ := any(complex(re, im)).(T)
		return result
	default:
		panic("unsupported complex type")
	}
}

// complexFromFloat64 is a private copy for backward compatibility.
func complexFromFloat64[T Complex](re, im float64) T {
	var zero T

	switch any(zero).(type) {
	case complex64:
		result, _ := any(complex(float32(re), float32(im))).(T)
		return result
	case complex128:
		result, _ := any(complex(re, im)).(T)
		return result
	default:
		panic("unsupported complex type")
	}
}

// Conj returns the complex conjugate of val.
func Conj[T Complex](val T) T {
	switch v := any(val).(type) {
	case complex64:
		return any(complex(real(v), -imag(v))).(T)
	case complex128:
		return any(complex(real(v), -imag(v))).(T)
	default:
		panic("unsupported complex type")
	}
}

// conj is a private wrapper for internal use.
func conj[T Complex](val T) T {
	switch v := any(val).(type) {
	case complex64:
		return any(complex(real(v), -imag(v))).(T)
	case complex128:
		return any(complex(real(v), -imag(v))).(T)
	default:
		panic("unsupported complex type")
	}
}

// ConjugateOf returns the complex conjugate of val.
// This is exported for use by the Plan type.
func ConjugateOf[T Complex](val T) T {
	return Conj(val)
}
