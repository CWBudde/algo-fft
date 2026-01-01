package fft

// ComplexMulArrayComplex64 computes element-wise complex multiplication: dst[i] = a[i] * b[i].
// All slices must have the same length.
// Uses SIMD acceleration when available.
func ComplexMulArrayComplex64(dst, a, b []complex64) {
	if !complexMulArrayComplex64SIMD(dst, a, b) {
		complexMulArrayComplex64Generic(dst, a, b)
	}
}

// ComplexMulArrayComplex128 computes element-wise complex multiplication: dst[i] = a[i] * b[i].
// All slices must have the same length.
// Uses SIMD acceleration when available.
func ComplexMulArrayComplex128(dst, a, b []complex128) {
	if !complexMulArrayComplex128SIMD(dst, a, b) {
		complexMulArrayComplex128Generic(dst, a, b)
	}
}

// ComplexMulArrayInPlaceComplex64 computes element-wise complex multiplication in-place: dst[i] *= src[i].
// Uses SIMD acceleration when available.
func ComplexMulArrayInPlaceComplex64(dst, src []complex64) {
	if !complexMulArrayInPlaceComplex64SIMD(dst, src) {
		complexMulArrayInPlaceComplex64Generic(dst, src)
	}
}

// ComplexMulArrayInPlaceComplex128 computes element-wise complex multiplication in-place: dst[i] *= src[i].
// Uses SIMD acceleration when available.
func ComplexMulArrayInPlaceComplex128(dst, src []complex128) {
	if !complexMulArrayInPlaceComplex128SIMD(dst, src) {
		complexMulArrayInPlaceComplex128Generic(dst, src)
	}
}

// Generic (pure Go) implementations.

func complexMulArrayComplex64Generic(dst, a, b []complex64) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayComplex128Generic(dst, a, b []complex128) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayInPlaceComplex64Generic(dst, src []complex64) {
	for i := range dst {
		dst[i] *= src[i]
	}
}

func complexMulArrayInPlaceComplex128Generic(dst, src []complex128) {
	for i := range dst {
		dst[i] *= src[i]
	}
}
