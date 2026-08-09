//go:build arm64 && !purego

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

//go:noescape
func Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, idx []int32, limit int, inverse bool, scale float32) bool

//go:noescape
func Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, idx []int32, limit int, inverse bool, scale float64) bool

// Size-specific complex64 NEON kernels.

//go:noescape
func ForwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

// NOTE: the complex64 32/128/512/2048/8192/32768 mixed-radix/radix-4-then-2
// kernels used to be declared here (backed by size-specific asm). They are
// now pure-Go wrappers over the shared NEON Stockham radix-4 core, extended
// to serve n = 2*4^k; see neon_radix4_loop.go.

// Size-specific complex128 NEON kernels.

//go:noescape
func ForwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// Size-specific NEON kernels for n = 2*4^k (32, 128, 512, 2048, 8192, 32768)
// are now thin Go wrappers over the shared radix-4 core in
// neon_radix4_loop_f64.go (see neon_f64_radix4_loop.s's "n = 2*4^k
// EXTENSION"), not standalone asm symbols, so they are not declared here.

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

// Inverse real FFT repack helpers. count is the number of k-bins processed
// in blocks of 2 (must be a multiple of 2, <= (len(dst)-1)/2); see
// neon_real_repack.s for the full contract.

//go:noescape
func InverseRepackComplex64NEONAsm(dst, src, weight []complex64, count int)

//go:noescape
func InverseRepackComplex128NEONAsm(dst, src, weight []complex128, count int)
