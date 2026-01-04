//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// registerAVX2DITCodelets64 registers AVX2-optimized complex64 DIT codelets.
// These registrations are conditional on the asm build tag and amd64 architecture.
func registerAVX2DITCodelets64() {
	// Size 4: Radix-4 AVX2 variant
	// Note: This implementation exists but may not provide speedup over generic
	// due to scalar operations. Registered with low priority for benchmarking.
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCodelet64(forwardDIT4Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT4Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit4_radix4_avx2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
	})

	// Size 8: Radix-8 AVX2 variant (single-stage butterfly)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size8Radix8Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size8Radix8Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix8_avx2",
		Priority:   9, // Keep below Go radix-8 unless proven faster
		BitrevFunc: mathpkg.ComputeIdentityIndices,
	})

	// Size 16: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size16Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix2_avx2",
		Priority:   20,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 16: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size16Radix4Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size16Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix4_avx2",
		Priority:   30,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 16: Radix-16 AVX2 variant (4x4)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size16Radix16Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size16Radix16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix16_avx2",
		Priority:   50, // Highest priority
		BitrevFunc: mathpkg.ComputeIdentityIndices,
	})

	// Size 32: Radix-2 AVX2 variant
	// TODO: This kernel has a bug causing incorrect inverse transforms.
	// Priority lowered to -1 to disable until fixed.
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size32Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size32Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix2_avx2",
		Priority:   -1, // Disabled: roundtrip test fails
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 64: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size64Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size64Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix2_avx2",
		Priority:   15,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 64: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(forwardDIT64Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT64Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix4_avx2",
		Priority:   25, // Prefer radix-4 for size 64
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 256: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size256Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size256Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix2_avx2",
		Priority:   15, // Lower than radix-4 variant
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 256: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(forwardDIT256Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT256Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix4_avx2",
		Priority:   20, // Higher priority - potentially faster
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 512: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size512Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size512Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_radix2_avx2",
		Priority:   10, // Baseline until a fully-unrolled kernel is available
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 512: Mixed-radix-2/4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCodelet64(amd64.ForwardAVX2Size512Mixed24Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseAVX2Size512Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_mixed24_avx2",
		Priority:   25,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesMixed24,
	})
}

// registerAVX2DITCodelets128 registers AVX2-optimized complex128 DIT codelets.
func registerAVX2DITCodelets128() {
	// Size 8: Radix-8 AVX2 variant (single-stage butterfly)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size8Radix8Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size8Radix8Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix8_avx2",
		Priority:   30, // Higher priority since it's proven faster
		BitrevFunc: nil,
	})

	// Size 8: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size8Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size8Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix2_avx2",
		Priority:   25, // Good fallback after radix-8
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 8: Radix-4 (Mixed-radix) AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size8Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size8Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix4_avx2",
		Priority:   11,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 16: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size16Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size16Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix2_avx2",
		Priority:   20,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 32: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size32Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size32Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix2_avx2",
		Priority:   20,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 64: Radix-2 AVX2 variant
	// TODO: This kernel has a bug causing incorrect inverse transforms.
	// Priority lowered to -1 to disable until fixed.
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size64Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size64Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix2_avx2",
		Priority:   -1, // Disabled: roundtrip test fails
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 64: Radix-4 AVX2 variant
	// TODO: This kernel has a bug causing incorrect inverse transforms.
	// Priority lowered to -1 to disable until fixed.
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size64Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size64Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix4_avx2",
		Priority:   -1, // Disabled: roundtrip test fails
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 128: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size128Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size128Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit128_radix2_avx2",
		Priority:   20,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 256: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size256Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size256Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix2_avx2",
		Priority:   20,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 512: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size512Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size512Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_radix2_avx2",
		Priority:   10, // Baseline until a fully-unrolled kernel is available
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 512: Mixed-radix-2/4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCodelet128(amd64.ForwardAVX2Size512Mixed24Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseAVX2Size512Mixed24Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_mixed24_avx2",
		Priority:   25,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesMixed24,
	})
}
