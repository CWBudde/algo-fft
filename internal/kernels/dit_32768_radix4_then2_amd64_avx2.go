//go:build amd64 && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// The AVX2 size-32768 kernel takes the mixed digit-reversal table as an
// argument (a 32768-entry DATA table would add 256 KiB to the binary), so these
// wrappers bind the shared table to the CodeletFunc signature.
//
// complex64 only has a 256-bit radix-4 kernel now (dit32768_radix4_avx2),
// which is ~2.8x faster, so only the complex128 side remains here.

func forwardDIT32768Radix4Then2AVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return amd64.ForwardAVX2Size32768Radix4Then2Complex128Asm(dst, src, twiddle, scratch, bitrevSize32768Radix4Then2)
}

func inverseDIT32768Radix4Then2AVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return amd64.InverseAVX2Size32768Radix4Then2Complex128Asm(dst, src, twiddle, scratch, bitrevSize32768Radix4Then2)
}
