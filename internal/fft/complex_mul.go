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

// ComplexMulArrayInPlace computes element-wise complex multiplication in-place
// (dst[i] *= src[i]) for the complex type T, dispatching to the SIMD-accelerated
// ComplexMulArrayInPlaceComplex64/128 entrypoints. The default branch is
// unreachable under the Complex constraint.
func ComplexMulArrayInPlace[T Complex](dst, src []T) {
	switch d := any(dst).(type) {
	case []complex64:
		ComplexMulArrayInPlaceComplex64(d, any(src).([]complex64))
	case []complex128:
		ComplexMulArrayInPlaceComplex128(d, any(src).([]complex128))
	default:
		complexMulArrayInPlaceGeneric(dst, src)
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
