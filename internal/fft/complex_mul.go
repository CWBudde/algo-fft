package fft

import m "github.com/cwbudde/algo-fft/internal/math"

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

// ComplexMulArray computes element-wise complex multiplication
// (dst[i] = a[i] * b[i]) for the complex type T, dispatching to the
// SIMD-accelerated ComplexMulArrayComplex64/128 entrypoints. All slices must
// have the same length. The default branch is unreachable under the Complex
// constraint.
func ComplexMulArray[T Complex](dst, a, b []T) {
	switch d := any(dst).(type) {
	case []complex64:
		ComplexMulArrayComplex64(d, any(a).([]complex64), any(b).([]complex64))
	case []complex128:
		ComplexMulArrayComplex128(d, any(a).([]complex128), any(b).([]complex128))
	default:
		complexMulArrayGeneric(dst, a, b)
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

// The complex64 fallbacks multiply through math.MulComplex64 rather than the
// `*` operator: Go compiles scalar complex64 multiplication by widening to
// complex128 and rounding back, which costs twice the instructions and makes
// the pure-Go path slower than its complex128 twin (see math.MulComplex64).
func complexMulArrayComplex64Generic(dst, a, b []complex64) {
	for i := range dst {
		dst[i] = m.MulComplex64(a[i], b[i])
	}
}

func complexMulArrayComplex128Generic(dst, a, b []complex128) {
	complexMulArrayGeneric(dst, a, b)
}

func complexMulArrayInPlaceComplex64Generic(dst, src []complex64) {
	for i := range dst {
		dst[i] = m.MulComplex64(dst[i], src[i])
	}
}

func complexMulArrayInPlaceComplex128Generic(dst, src []complex128) {
	complexMulArrayInPlaceGeneric(dst, src)
}
