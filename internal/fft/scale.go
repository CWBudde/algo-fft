package fft

// ScaleComplex64InPlace scales each element in dst by scale.
// Uses SIMD acceleration when available.
func ScaleComplex64InPlace(dst []complex64, scale float32) {
	if scale == 1 {
		return
	}

	if !scaleComplex64SIMD(dst, scale) {
		factor := complex(scale, 0)
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
