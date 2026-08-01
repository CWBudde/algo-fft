//go:build arm64 && !purego

package arm64

// This file replaces five fully-unrolled scalar "NEON" codelets (sizes 64,
// 256, 1024, 4096 and 16384 — about 28,500 lines of assembly containing no
// vector instructions at all) with thin wrappers over a single looped,
// genuinely vectorized Stockham radix-4 core. The exported names and
// signatures are unchanged, so the codelet specs and the generated registry
// initializers keep working untouched.
//
// The core is in neon_f32_radix4_loop.s; see its header for the algorithm,
// the two vectorization regimes and the scratch/dst buffering rules.

//go:noescape
func neonRadix4ForwardC64(dst, src, twiddle, scratch []complex64, n int) bool

//go:noescape
func neonRadix4InverseC64(dst, src, twiddle, scratch []complex64, n int, scale float32) bool

// ForwardNEONSize64Radix2Complex64Asm computes a size-64 forward complex64
// FFT. Despite the historical "Radix2" in its name it runs the shared NEON
// Stockham radix-4 core. It requires src of exactly 64 elements and dst,
// twiddle and scratch of at least 64, and returns false otherwise.
func ForwardNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4ForwardC64(dst, src, twiddle, scratch, 64)
}

// InverseNEONSize64Radix2Complex64Asm computes a size-64 inverse complex64
// FFT, normalized by 1/64. Despite the historical "Radix2" in its name it runs
// the shared NEON Stockham radix-4 core. It requires src of exactly 64
// elements and dst, twiddle and scratch of at least 64, and returns false
// otherwise.
func InverseNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4InverseC64(dst, src, twiddle, scratch, 64, 1.0/64.0)
}

// ForwardNEONSize256Radix4Complex64Asm computes a size-256 forward complex64
// FFT with the shared NEON Stockham radix-4 core. It requires src of exactly
// 256 elements and dst, twiddle and scratch of at least 256, and returns false
// otherwise.
func ForwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4ForwardC64(dst, src, twiddle, scratch, 256)
}

// InverseNEONSize256Radix4Complex64Asm computes a size-256 inverse complex64
// FFT, normalized by 1/256, with the shared NEON Stockham radix-4 core. It
// requires src of exactly 256 elements and dst, twiddle and scratch of at
// least 256, and returns false otherwise.
func InverseNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4InverseC64(dst, src, twiddle, scratch, 256, 1.0/256.0)
}

// ForwardNEONSize1024Radix4Complex64Asm computes a size-1024 forward complex64
// FFT with the shared NEON Stockham radix-4 core. It requires src of exactly
// 1024 elements and dst, twiddle and scratch of at least 1024, and returns
// false otherwise.
func ForwardNEONSize1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4ForwardC64(dst, src, twiddle, scratch, 1024)
}

// InverseNEONSize1024Radix4Complex64Asm computes a size-1024 inverse complex64
// FFT, normalized by 1/1024, with the shared NEON Stockham radix-4 core. It
// requires src of exactly 1024 elements and dst, twiddle and scratch of at
// least 1024, and returns false otherwise.
func InverseNEONSize1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4InverseC64(dst, src, twiddle, scratch, 1024, 1.0/1024.0)
}

// ForwardNEONSize4096Radix4Complex64Asm computes a size-4096 forward complex64
// FFT with the shared NEON Stockham radix-4 core. It requires src of exactly
// 4096 elements and dst, twiddle and scratch of at least 4096, and returns
// false otherwise.
func ForwardNEONSize4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4ForwardC64(dst, src, twiddle, scratch, 4096)
}

// InverseNEONSize4096Radix4Complex64Asm computes a size-4096 inverse complex64
// FFT, normalized by 1/4096, with the shared NEON Stockham radix-4 core. It
// requires src of exactly 4096 elements and dst, twiddle and scratch of at
// least 4096, and returns false otherwise.
func InverseNEONSize4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4InverseC64(dst, src, twiddle, scratch, 4096, 1.0/4096.0)
}

// ForwardNEONSize16384Radix4Complex64Asm computes a size-16384 forward
// complex64 FFT with the shared NEON Stockham radix-4 core. It requires src of
// exactly 16384 elements and dst, twiddle and scratch of at least 16384, and
// returns false otherwise.
func ForwardNEONSize16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4ForwardC64(dst, src, twiddle, scratch, 16384)
}

// InverseNEONSize16384Radix4Complex64Asm computes a size-16384 inverse
// complex64 FFT, normalized by 1/16384, with the shared NEON Stockham radix-4
// core. It requires src of exactly 16384 elements and dst, twiddle and scratch
// of at least 16384, and returns false otherwise.
func InverseNEONSize16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return neonRadix4InverseC64(dst, src, twiddle, scratch, 16384, 1.0/16384.0)
}
