//go:build arm64 && asm && !purego

package arm64

// NOTE: These are Go declarations for ARM64 assembly routines implemented in the *.s files in this directory.

//go:noescape
func ForwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONComplex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONComplex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONComplex128Asm(dst, src, twiddle, scratch []complex128) bool

// Size-specific complex64 NEON kernels.

//go:noescape
func ForwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

// Size-specific complex128 NEON kernels.

//go:noescape
func ForwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize16Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize16Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize32Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize32Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize256Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize256Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// Complex multiply helpers.

//go:noescape
func ComplexMulArrayComplex64NEONAsm(dst, a, b []complex64)

//go:noescape
func ComplexMulArrayInPlaceComplex64NEONAsm(dst, src []complex64)

//go:noescape
func ComplexMulArrayComplex128NEONAsm(dst, a, b []complex128)

//go:noescape
func ComplexMulArrayInPlaceComplex128NEONAsm(dst, src []complex128)

// Complex array scaling (element-wise) with scalar factors.

//go:noescape
func ScaleComplex64NEONAsm(dst []complex64, scale float32)

//go:noescape
func ScaleComplex128NEONAsm(dst []complex128, scale float64)

// Inverse real FFT repack helpers (complex64 only).

//go:noescape
func InverseRepackComplex64NEONAsm(dst, src, weight []complex64, kStartMax int)
