//go:build amd64 && fft_asm

package fft

func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch {
	case len(dst) == 1 && len(src) == 1:
		asmCopyComplex64(&dst[0], &src[0])
		return true
	case len(dst) == 2 && len(src) == 2:
		return asmForward2Complex64(&dst[0], &src[0])
	default:
		return false
	}
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch {
	case len(dst) == 1 && len(src) == 1:
		asmCopyComplex64(&dst[0], &src[0])
		return true
	case len(dst) == 2 && len(src) == 2:
		return asmInverse2Complex64(&dst[0], &src[0])
	default:
		return false
	}
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardAVX2Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseAVX2Complex64(dst, src, twiddle, scratch, bitrev)
}
