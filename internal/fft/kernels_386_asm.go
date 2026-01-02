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
	return auto
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	// No complex128 SSE2 implementation for 386 yet, use pure Go
	return autoKernelComplex128(KernelAuto)
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}
	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	// No complex128 SSE2 implementation for 386 yet, use pure Go
	return autoKernelComplex128(strategy)
}

// SSE2 wrapper functions for complex64 (delegate to asm_386.go imports)

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
