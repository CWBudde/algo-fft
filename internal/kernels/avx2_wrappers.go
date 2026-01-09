//go:build amd64 && asm && !purego

package kernels

import amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"

// forwardAVX2Size256Radix4Complex64Safe ensures in-place forward uses dst output.
// The AVX2 radix-4 size-256 forward kernel writes to scratch for in-place use
// but does not copy results back to dst.
func forwardAVX2Size256Radix4Complex64Safe(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256
	if len(dst) < n || len(src) < n || len(scratch) < n {
		return amd64.ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}

	if &dst[0] == &src[0] {
		ok := amd64.ForwardAVX2Size256Radix4Complex64Asm(scratch, src, twiddle, scratch, bitrev)
		if ok {
			copy(dst[:n], scratch[:n])
		}
		return ok
	}

	return amd64.ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
