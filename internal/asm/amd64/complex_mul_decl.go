//go:build amd64

package amd64

// Complex array multiplication (element-wise) - AVX2 optimized.
// These functions are available on any amd64 platform, with runtime
// CPU feature detection for optimal path selection.

//go:noescape
func ComplexMulArrayComplex64AVX2Asm(dst, a, b []complex64)

//go:noescape
func ComplexMulArrayInPlaceComplex64AVX2Asm(dst, src []complex64)

//go:noescape
func ComplexMulArrayComplex128AVX2Asm(dst, a, b []complex128)

//go:noescape
func ComplexMulArrayInPlaceComplex128AVX2Asm(dst, src []complex128)
