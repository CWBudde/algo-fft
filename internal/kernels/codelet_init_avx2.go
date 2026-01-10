//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// registerAVX2DITCodelets64 registers AVX2-optimized complex64 DIT codelets.
// These registrations are conditional on the asm build tag and amd64 architecture.
func registerAVX2DITCodelets64() {
	// Size 4: Radix-4 AVX2 variant
	// Note: This implementation exists but may not provide speedup over generic
	// due to scalar operations. Registered with low priority for benchmarking.
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCore64(amd64.ForwardAVX2Size4Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size4Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit4_radix4_avx2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained, no external bitrev
	})

	// Size 8: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(amd64.ForwardAVX2Size8Radix2Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size8Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix2_avx2",
		Priority:   7,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained with internalized bitrev
	})

	// Size 8: Radix-4 (Mixed-radix) AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(amd64.ForwardAVX2Size8Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size8Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix4_avx2",
		Priority:   10, // Mixed-radix 4x2: efficient for size 8
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained with internalized bitrev
	})

	// Size 8: Radix-8 AVX2 variant (single-stage butterfly, identity permutation)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(amd64.ForwardAVX2Size8Radix8Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size8Radix8Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix8_avx2",
		Priority:   9, // Keep below Go radix-8 unless proven faster
		BitrevFunc: nil,
		KernelType: KernelTypeCore, // Identity permutation, no bitrev needed
	})

	// Size 16: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(amd64.ForwardAVX2Size16Radix2Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size16Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix2_avx2",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(amd64.ForwardAVX2Size16Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size16Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix4_avx2",
		Priority:   30,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-16 AVX2 variant (4x4)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(amd64.ForwardAVX2Size16Radix16Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size16Radix16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix16_avx2",
		Priority:   50, // Highest priority
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Radix-32 AVX2 variant (4x8 factorization, identity permutation)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(amd64.ForwardAVX2Size32Radix32Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size32Radix32Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix32_avx2",
		Priority:   25, // Higher than radix-2 variants
		BitrevFunc: nil,
		KernelType: KernelTypeCore, // Identity permutation, no bitrev needed
	})

	// Size 32: Radix-2 AVX2 variant
	// Uses 5-stage unrolled DIT with bit-reversal permutation
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(amd64.ForwardAVX2Size32Radix2Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size32Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix2_avx2",
		Priority:   20, // Below radix-32 (25), above scalar radix-2 (17)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardAVX2Size64Complex64Asm, bitrevSize64Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseAVX2Size64Complex64Asm, bitrevSize64Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix2_avx2",
		Priority:   19, // Below radix-4 (25), above scalar radix-2 (17)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(amd64.ForwardAVX2Size64Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size64Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix4_avx2",
		Priority:   25, // Prefer radix-4 for size 64
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Mixed-radix-2/4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(amd64.ForwardAVX2Size128Mixed24Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size128Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit128_mixed24_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(amd64.ForwardAVX2Size256Radix2Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size256Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix2_avx2",
		Priority:   15, // Lower than radix-4 variant
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(forwardAVX2Size256Radix4Complex64Safe),
		Inverse:    wrapCore64(amd64.InverseAVX2Size256Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix4_avx2",
		Priority:   120, // Higher priority than generic
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-16 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(amd64.ForwardAVX2Size256Radix16Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size256Radix16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix16_avx2",
		Priority:   130, // Highest priority (1.5x faster than radix-4)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardAVX2Size512Radix2Complex64Asm, bitrevSize512Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseAVX2Size512Radix2Complex64Asm, bitrevSize512Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_radix2_avx2",
		Priority:   10, // Baseline until a fully-unrolled kernel is available
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Mixed-radix-2/4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(amd64.ForwardAVX2Size512Mixed24Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size512Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_mixed24_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Radix-8 AVX2 variant (3-stage unrolled)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(amd64.ForwardAVX2Size512Radix8Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size512Radix8Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_radix8_avx2",
		Priority:   30, // Higher than mixed-2/4 (25)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Radix-16x32 AVX2 variant (six-step, identity permutation)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(amd64.ForwardAVX2Size512Radix16x32Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size512Radix16x32Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_radix16x32_avx2",
		Priority:   20,  // Between mixed-2/4 (25) and radix-2 (10)
		BitrevFunc: nil, // Kernel handles permutation internally (no external bitrev)
		KernelType: KernelTypeDIT,
	})

	// Size 384: Mixed-radix (128×3) variant
	// Decomposed as radix-3 + 128-point sub-FFTs
	// BitrevFunc is nil because the composite algorithm handles permutation internally
	Registry64.Register(CodeletEntry[complex64]{
		Size:       384,
		Forward:    wrapCore64(forwardDIT384MixedComplex64),
		Inverse:    wrapCore64(inverseDIT384MixedComplex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit384_mixed_avx2",
		Priority:   25,
		BitrevFunc: nil, // Composite algorithm - no external bitrev needed
		KernelType: KernelTypeDIT,
	})

	// Size 2048: Mixed-radix-2/4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       2048,
		Forward:    wrapCore64(amd64.ForwardAVX2Size2048Mixed24Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size2048Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit2048_mixed24_avx2",
		Priority:   -1, // Disabled: roundtrip failures under asm tag (see PLAN.md)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 8192: Mixed-radix-2/4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8192,
		Forward:    wrapCore64(amd64.ForwardAVX2Size8192Mixed24Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size8192Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8192_mixed24_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 1024: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       1024,
		Forward:    wrapCore64(amd64.ForwardAVX2Size1024Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size1024Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit1024_radix4_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 4096: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4096,
		Forward:    wrapCore64(amd64.ForwardAVX2Size4096Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size4096Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit4096_radix4_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 4096: Six-step (64×64) AVX2 variant
	// Uses the six-step algorithm with AVX2-optimized 64-point sub-FFTs
	// ~30-40% faster than radix-4 due to better cache utilization
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4096,
		Forward:    wrapCore64(forwardDIT4096SixStepAVX2Complex64),
		Inverse:    wrapCore64(inverseDIT4096SixStepAVX2Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit4096_sixstep_avx2",
		Priority:   35, // Higher priority than radix-4 (25)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16384: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16384,
		Forward:    wrapCore64(amd64.ForwardAVX2Size16384Radix4Complex64Asm),
		Inverse:    wrapCore64(amd64.InverseAVX2Size16384Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16384_radix4_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16384: Six-step (128×128) AVX2 variant
	// Uses the six-step algorithm with AVX2-optimized 128-point sub-FFTs
	// Better cache utilization for large transforms
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16384,
		Forward:    wrapCore64(forwardDIT16384SixStepAVX2Complex64),
		Inverse:    wrapCore64(inverseDIT16384SixStepAVX2Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16384_sixstep_avx2",
		Priority:   35, // Higher priority than radix-4 (25)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}

// registerAVX2DITCodelets128 registers AVX2-optimized complex128 DIT codelets.
func registerAVX2DITCodelets128() {
	// Size 384: Mixed-radix (128×3) variant
	// Decomposed as radix-3 + 128-point sub-FFTs
	// BitrevFunc is nil because the composite algorithm handles permutation internally
	Registry128.Register(CodeletEntry[complex128]{
		Size:       384,
		Forward:    wrapCore128(forwardDIT384MixedComplex128),
		Inverse:    wrapCore128(inverseDIT384MixedComplex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit384_mixed_avx2",
		Priority:   25,
		BitrevFunc: nil, // Composite algorithm - no external bitrev needed
		KernelType: KernelTypeDIT,
	})

	// Size 4: Radix-4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4,
		Forward:    wrapCore128(forwardDIT4Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT4Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit4_radix4_avx2",
		Priority:   5,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Updated from Core to DIT consistent with 64-bit
	})

	// Size 8: Radix-8 AVX2 variant (single-stage butterfly, identity permutation)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(amd64.ForwardAVX2Size8Radix8Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size8Radix8Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix8_avx2",
		Priority:   9, // Keep below generic radix-8 until proven faster
		BitrevFunc: nil,
		KernelType: KernelTypeCore, // Identity permutation, no bitrev needed
	})

	// Size 8: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(amd64.ForwardAVX2Size8Radix2Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size8Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix2_avx2",
		Priority:   25, // Good fallback after radix-8
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 8: Radix-4 (Mixed-radix) AVX2 variant
	// Uses mixed-radix algorithm: radix-4 stage followed by radix-2 stage
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(amd64.ForwardAVX2Size8Radix4Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size8Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix4_avx2",
		Priority:   30, // Preferred over radix-8 and radix-2
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCore128(amd64.ForwardAVX2Size16Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size16Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix2_avx2",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCore128(amd64.ForwardAVX2Size16Radix4Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size16Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix4_avx2",
		Priority:   30,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCore128(amd64.ForwardAVX2Size32Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size32Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix2_avx2",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(amd64.ForwardAVX2Size64Radix2Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size64Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix2_avx2",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardAVX2Size64Radix4Complex128Asm, bitrevSize64Radix4)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseAVX2Size64Radix4Complex128Asm, bitrevSize64Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix4_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCore128(amd64.ForwardAVX2Size128Radix2Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size128Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit128_radix2_avx2",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCore128(amd64.ForwardAVX2Size256Radix2Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size256Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix2_avx2",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(amd64.ForwardAVX2Size512Radix2Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size512Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_radix2_avx2",
		Priority:   10, // Baseline until a fully-unrolled kernel is available
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Mixed-radix-2/4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(amd64.ForwardAVX2Size512Mixed24Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseAVX2Size512Mixed24Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit512_mixed24_avx2",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}
