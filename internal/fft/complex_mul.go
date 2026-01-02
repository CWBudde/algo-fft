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

func complexMulArrayGeneric[T Complex](dst, a, b []T) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func complexMulArrayInPlaceGeneric[T Complex](dst, src []T) {
	for i := range dst {
		dst[i] *= src[i]
	}
}

func complexMulArrayComplex64Generic(dst, a, b []complex64) {
	complexMulArrayGeneric(dst, a, b)
}

func complexMulArrayComplex128Generic(dst, a, b []complex128) {
	complexMulArrayGeneric(dst, a, b)
}

func complexMulArrayInPlaceComplex64Generic(dst, src []complex64) {
	complexMulArrayInPlaceGeneric(dst, src)
}

func complexMulArrayInPlaceComplex128Generic(dst, src []complex128) {
	complexMulArrayInPlaceGeneric(dst, src)
}
