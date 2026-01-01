//go:build amd64 && fft_asm && !purego

package kernels

import kasm "github.com/MeKo-Christian/algo-fft/internal/kernels/asm"

func forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT4Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT4Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT64Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT64Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return forwardDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return inverseDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(8)
	}
	return forwardDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(8)
	}
	return inverseDIT8Radix2Complex128(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

// SSE2 size-specific kernels
func forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return kasm.InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
