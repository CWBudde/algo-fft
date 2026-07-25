package fft

// ScaleComplex64InPlace scales each element in dst by scale.
// Uses SIMD acceleration when available.
func ScaleComplex64InPlace(dst []complex64, scale float32) {
	if scale == 1 {
		return
	}

	if !scaleComplex64SIMD(dst, scale) {
		// Scaling by a real factor component-wise rather than by a complex64
		// one: the complex multiply would widen every element to complex128
		// and round back (see math.MulComplex64), for the same two products.
		for i := range dst {
			dst[i] = complex(real(dst[i])*scale, imag(dst[i])*scale)
		}
	}
}

// ScaleInPlace scales each element of dst by the real factor scale,
// dispatching to the SIMD-accelerated ScaleComplex64/128InPlace entrypoints.
// The default branch is unreachable under the Complex constraint.
func ScaleInPlace[T Complex](dst []T, scale float64) {
	switch d := any(dst).(type) {
	case []complex64:
		ScaleComplex64InPlace(d, float32(scale))
	case []complex128:
		ScaleComplex128InPlace(d, scale)
	default:
		factor := complexFromFloat64[T](scale, 0)
		for i := range dst {
			dst[i] *= factor
		}
	}
}

// ScaleComplex128InPlace scales each element in dst by scale.
// Uses SIMD acceleration when available.
func ScaleComplex128InPlace(dst []complex128, scale float64) {
	if scale == 1 {
		return
	}

	if !scaleComplex128SIMD(dst, scale) {
		factor := complex(scale, 0)
		for i := range dst {
			dst[i] *= factor
		}
	}
}
