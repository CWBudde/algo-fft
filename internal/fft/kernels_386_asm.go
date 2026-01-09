//go:build 386 && asm && !purego

package fft

import "github.com/MeKo-Christian/algo-fft/internal/cpu"

// SSE2-only kernel selection for 386 architecture.
// 386 has SSE2 but not AVX2, so we use SSE2 kernels with pure-Go fallback.

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}
	if features.HasSSE && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSEComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSEComplex64, auto.Inverse),
		}
	}
	return auto
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128, auto.Inverse),
		}
	}
	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}
	if features.HasSSE && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSEComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSEComplex64, auto.Inverse),
		}
	}
	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128, auto.Inverse),
		}
	}
	return auto
}

// SSE (SSE1) wrapper functions for complex64

func forwardSSEComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 4:
		return forwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
	return forwardSSEComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSEComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 4:
		return inverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
	return inverseSSEComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

// SSE2 wrapper functions for complex64 (delegate to asm_386.go imports)

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 2:
		return forwardSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 4:
		return forwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 8:
		return forwardSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		// TODO(386): Re-enable size-16 radix-16 once x86 kernel is corrected.
	}
	return forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 2:
		return inverseSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 4:
		return inverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 8:
		return inverseSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		// TODO(386): Re-enable size-16 radix-16 once x86 kernel is corrected.
	}
	return inverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

// SSE2 wrapper functions for complex128

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	switch len(src) {
	case 2:
		return forwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
	case 4:
		return forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
	case 8:
		return forwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
	case 16:
		return forwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
	}
	return false // No generic SSE2 complex128 kernel yet
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	switch len(src) {
	case 2:
		return inverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
	case 4:
		return inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
	case 8:
		return inverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev)
	case 16:
		return inverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev)
	}
	return false
}
