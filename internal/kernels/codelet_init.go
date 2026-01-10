package kernels

import (
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// Type aliases for planner codelet types.
type (
	CodeletEntry[T Complex] = planner.CodeletEntry[T]
	CodeletFunc[T Complex]  = planner.CodeletFunc[T]
	SIMDLevel               = planner.SIMDLevel
)

// Re-export planner registries and SIMD constants.
var (
	Registry64  = planner.Registry64
	Registry128 = planner.Registry128
)

const (
	SIMDNone   = planner.SIMDNone
	SIMDSSE2   = planner.SIMDSSE2
	SIMDAVX2   = planner.SIMDAVX2
	SIMDAVX512 = planner.SIMDAVX512
	SIMDNEON   = planner.SIMDNEON
)

// KernelType constants for codelet classification.
const (
	KernelTypeLegacy = planner.KernelTypeLegacy
	KernelTypeCore   = planner.KernelTypeCore
	KernelTypeDIT    = planner.KernelTypeDIT
)

// GetRegistry returns the appropriate registry for type T.
func GetRegistry[T Complex]() *planner.CodeletRegistry[T] {
	return planner.GetRegistry[T]()
}

// This file registers all built-in codelets with the global registries.
// Registration happens at init time so codelets are available when plans are created.

//nolint:gochecknoinits
func init() {
	// Register complex64 DIT codelets
	registerDITCodelets64()

	// Register complex128 DIT codelets
	registerDITCodelets128()

	// Register NEON codelets (conditional on build tags)
	registerNEONDITCodelets64()
	registerNEONDITCodelets128()

	// Register SSE2 codelets (conditional on build tags)
	registerSSE2DITCodelets64()
	registerSSE2DITCodelets128()

	// Register AVX2 codelets (conditional on build tags)
	registerAVX2DITCodelets64()
	registerAVX2DITCodelets128()
}

// registerDITCodelets64 registers all complex64 DIT codelets with multiple radix variants.
func registerDITCodelets64() {
	// Size 4: Only radix-4 (no bit-reversal needed)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCore64(forwardDIT4Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT4Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4_radix4_generic",
		Priority:   0,
		BitrevFunc: nil,            // No bit-reversal for size 4
		KernelType: KernelTypeCore, // Self-contained, no external bitrev
	})

	// Size 8: Radix-2 variant (default for now)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(forwardDIT8Radix2Complex64),
		Inverse:    wrapCore64(inverseDIT8Radix2Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix2_generic",
		Priority:   20,
		BitrevFunc: nil,           // Internalized bit-reversal
		KernelType: KernelTypeDIT, // Self-contained with internal permutation
	})

	// Size 8: Radix-8 variant (single butterfly)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(forwardDIT8Radix8Complex64),
		Inverse:    wrapCore64(inverseDIT8Radix8Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix8_generic",
		Priority:   30,             // Highest priority among generic size-8 codelets
		BitrevFunc: nil,            // Radix-8 on size 8 is identity; internalize it
		KernelType: KernelTypeCore, // Self-contained, no external bitrev
	})

	// Size 8: Mixed-radix variant (1x radix-4 + 1x radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(forwardDIT8Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT8Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_mixedradix_generic",
		Priority:   25, // Between radix-2 (20) and radix-8 (30)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained with internal permutation
	})

	// Size 16: Radix-2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(forwardDIT16Radix2Complex64),
		Inverse:    wrapCore64(inverseDIT16Radix2Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix2_generic",
		Priority:   10,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained
	})

	// Size 16: Radix-4 variant (12-15% faster per benchmarks)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(forwardDIT16Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT16Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix4_generic",
		Priority:   20, // Higher priority (empirically faster)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained
	})

	// Size 32: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(forwardDIT32Complex64),
		Inverse:    wrapCore64(inverseDIT32Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_radix2_generic",
		Priority:   0,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(forwardDIT32MixedRadix24Complex64),
		Inverse:    wrapCore64(inverseDIT32MixedRadix24Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_mixed24_generic",
		Priority:   10, // Higher priority than plain radix-2
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(forwardDIT64Complex64),
		Inverse:    wrapCore64(inverseDIT64Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix2_generic",
		Priority:   0,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(forwardDIT64Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT64Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix4_generic",
		Priority:   20, // Higher priority (expected faster)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(forwardDIT128Complex64),
		Inverse:    wrapCore64(inverseDIT128Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_radix2_generic",
		Priority:   0,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(forwardDIT128MixedRadix24Complex64),
		Inverse:    wrapCore64(inverseDIT128MixedRadix24Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(forwardDIT256Complex64),
		Inverse:    wrapCore64(inverseDIT256Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix2_generic",
		Priority:   10,
	})

	// Size 256: Radix-4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(forwardDIT256Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT256Radix4Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix4_generic",
		Priority:   20, // Higher priority (potentially faster)
	})

	// Size 256: Radix-16 variant (16x16 decomposition)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(forwardDIT256Radix16Complex64),
		Inverse:    wrapCore64(inverseDIT256Radix16Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix16_generic",
		Priority:   30, // Optimized pure-Go implementation (beating radix-4)
	})

	// Size 512: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(forwardDIT512Complex64),
		Inverse:    wrapCore64(inverseDIT512Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix2_generic",
		Priority:   10,
	})

	// Size 512: Radix-8 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(forwardDIT512Radix8Complex64),
		Inverse:    wrapCore64(inverseDIT512Radix8Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix8_generic",
		Priority:   30,
	})

	// Size 512: Mixed-radix-2/4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(forwardDIT512Mixed24Complex64),
		Inverse:    wrapCore64(inverseDIT512Mixed24Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_mixed24_generic",
		Priority:   20,
	})

	// Size 512: Radix-16x32 variant (Six-step)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(forwardDIT512Mixed16x32Complex64),
		Inverse:    wrapCore64(inverseDIT512Mixed16x32Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix16x32_generic",
		Priority:   40,
	})

	// Size 1024: Radix-4 variant (1024 = 4^5, 5 stages vs 10 for radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       1024,
		Forward:    wrapCore64(forwardDIT1024Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT1024Radix4Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit1024_radix4_generic",
		Priority:   15,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       1024,
		Forward:    wrapCore64(forwardDIT1024Mixed32x32Complex64),
		Inverse:    wrapCore64(inverseDIT1024Mixed32x32Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit1024_radix32x32_generic",
		Priority:   25, // Usually faster than radix-4 for size 1024
	})

	// Size 2048: Mixed-radix-2/4 variant (odd log2, faster than pure radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       2048,
		Forward:    wrapCore64(forwardDIT2048Mixed24Complex64),
		Inverse:    wrapCore64(inverseDIT2048Mixed24Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit2048_mixedradix24_generic",
		Priority:   20,
	})

	// Size 4096: Radix-4 variant (4096 = 4^6, 6 stages vs 12 for radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4096,
		Forward:    wrapCore64(forwardDIT4096Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT4096Radix4Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4096_radix4_generic",
		Priority:   15,
	})

	// Size 4096: Six-Step variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4096,
		Forward:    wrapCore64(forwardDIT4096SixStepComplex64),
		Inverse:    wrapCore64(inverseDIT4096SixStepComplex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4096_sixstep_generic",
		Priority:   25,
	})

	// Size 8192: Mixed-radix-2/4 variant (odd log2, faster than pure radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8192,
		Forward:    wrapCore64(forwardDIT8192Mixed24Complex64),
		Inverse:    wrapCore64(inverseDIT8192Mixed24Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8192_mixedradix24_generic",
		Priority:   20,
	})

	// Size 16384: Radix-4 variant (16384 = 4^7, 7 stages vs 14 for radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16384,
		Forward:    wrapCore64(forwardDIT16384Radix4Complex64),
		Inverse:    wrapCore64(inverseDIT16384Radix4Complex64),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16384_radix4_generic",
		Priority:   15,
	})
}

// registerDITCodelets128 registers all complex128 DIT codelets with multiple radix variants.
func registerDITCodelets128() {
	// Size 4: Only radix-4 (no bit-reversal needed)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4,
		Forward:    wrapCore128(forwardDIT4Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT4Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4_radix4_generic",
		Priority:   0,
		BitrevFunc: nil,            // No bit-reversal for size 4
		KernelType: KernelTypeCore, // Self-contained, no external bitrev
	})

	// Size 8: Radix-2 variant (default for now)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(forwardDIT8Radix2Complex128),
		Inverse:    wrapCore128(inverseDIT8Radix2Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix2_generic",
		Priority:   20,
		BitrevFunc: nil,           // Internalized bit-reversal
		KernelType: KernelTypeDIT, // Self-contained
	})

	// Size 8: Radix-8 variant (single butterfly)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(forwardDIT8Radix8Complex128),
		Inverse:    wrapCore128(inverseDIT8Radix8Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix8_generic",
		Priority:   30,             // Highest priority
		BitrevFunc: nil,            // Radix-8 on size 8 is identity; internalize it
		KernelType: KernelTypeCore, // Self-contained
	})

	// Size 8: Mixed-radix variant (1x radix-4 + 1x radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(forwardDIT8Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT8Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_mixedradix_generic",
		Priority:   25, // Between radix-2 (20) and radix-8 (30)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained with internal permutation
	})

	// Size 16: Radix-2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCore128(forwardDIT16Radix2Complex128),
		Inverse:    wrapCore128(inverseDIT16Radix2Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix2_generic",
		Priority:   10,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained
	})

	// Size 16: Radix-4 variant (12-15% faster per benchmarks)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCore128(forwardDIT16Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT16Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix4_generic",
		Priority:   20, // Higher priority (empirically faster)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained
	})

	// Size 32: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCore128(forwardDIT32Complex128),
		Inverse:    wrapCore128(inverseDIT32Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_radix2_generic",
		Priority:   0,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCore128(forwardDIT32MixedRadix24Complex128),
		Inverse:    wrapCore128(inverseDIT32MixedRadix24Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(forwardDIT64Complex128),
		Inverse:    wrapCore128(inverseDIT64Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix2_generic",
		Priority:   0,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(forwardDIT64Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT64Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix4_generic",
		Priority:   20, // Higher priority (expected faster)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCore128(forwardDIT128Complex128),
		Inverse:    wrapCore128(inverseDIT128Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_radix2_generic",
		Priority:   0,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCore128(forwardDIT128MixedRadix24Complex128),
		Inverse:    wrapCore128(inverseDIT128MixedRadix24Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCore128(forwardDIT256Complex128),
		Inverse:    wrapCore128(inverseDIT256Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix2_generic",
		Priority:   10,
	})

	// Size 256: Radix-4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCore128(forwardDIT256Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT256Radix4Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix4_generic",
		Priority:   20, // Higher priority (potentially faster)
	})

	// Size 256: Radix-16 variant (16x16 decomposition)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCore128(forwardDIT256Radix16Complex128),
		Inverse:    wrapCore128(inverseDIT256Radix16Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix16_generic",
		Priority:   15, // Pure-Go implementation (currently slower than radix-4)
	})

	// Size 512: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(forwardDIT512Complex128),
		Inverse:    wrapCore128(inverseDIT512Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix2_generic",
		Priority:   10,
	})

	// Size 512: Radix-8 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(forwardDIT512Radix8Complex128),
		Inverse:    wrapCore128(inverseDIT512Radix8Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix8_generic",
		Priority:   30,
	})

	// Size 512: Mixed-radix-2/4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(forwardDIT512Mixed24Complex128),
		Inverse:    wrapCore128(inverseDIT512Mixed24Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_mixed24_generic",
		Priority:   20,
	})

	// Size 512: Radix-16x32 variant (Six-step)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(forwardDIT512Mixed16x32Complex128),
		Inverse:    wrapCore128(inverseDIT512Mixed16x32Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix16x32_generic",
		Priority:   40,
	})

	// Size 1024: Radix-4 variant (1024 = 4^5, 5 stages vs 10 for radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       1024,
		Forward:    wrapCore128(forwardDIT1024Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT1024Radix4Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit1024_radix4_generic",
		Priority:   15,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       1024,
		Forward:    wrapCore128(forwardDIT1024Mixed32x32Complex128),
		Inverse:    wrapCore128(inverseDIT1024Mixed32x32Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit1024_radix32x32_generic",
		Priority:   25,
	})

	// Size 2048: Mixed-radix-2/4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       2048,
		Forward:    wrapCore128(forwardDIT2048Mixed24Complex128),
		Inverse:    wrapCore128(inverseDIT2048Mixed24Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit2048_mixedradix24_generic",
		Priority:   20,
	})

	// Size 4096: Radix-4 variant (4096 = 4^6, 6 stages vs 12 for radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4096,
		Forward:    wrapCore128(forwardDIT4096Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT4096Radix4Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4096_radix4_generic",
		Priority:   15,
	})

	// Size 4096: Six-Step variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4096,
		Forward:    wrapCore128(forwardDIT4096SixStepComplex128),
		Inverse:    wrapCore128(inverseDIT4096SixStepComplex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4096_sixstep_generic",
		Priority:   25,
	})

	// Size 8192: Mixed-radix-2/4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8192,
		Forward:    wrapCore128(forwardDIT8192Mixed24Complex128),
		Inverse:    wrapCore128(inverseDIT8192Mixed24Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8192_mixedradix24_generic",
		Priority:   20,
	})

	// Size 16384: Radix-4 variant (16384 = 4^7, 7 stages vs 14 for radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16384,
		Forward:    wrapCore128(forwardDIT16384Radix4Complex128),
		Inverse:    wrapCore128(inverseDIT16384Radix4Complex128),
		Algorithm:  KernelDIT,
		KernelType: KernelTypeDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16384_radix4_generic",
		Priority:   15,
	})
}

// KernelFunc64 is the signature of existing complex64 kernels that return bool.
type KernelFunc64 func(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// KernelFunc128 is the signature of existing complex128 kernels that return bool.
type KernelFunc128 func(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// CoreKernelFunc64 is the signature of refactored complex64 kernels without bitrev.
type CoreKernelFunc64 func(dst, src, twiddle, scratch []complex64) bool

// CoreKernelFunc128 is the signature of refactored complex128 kernels without bitrev.
type CoreKernelFunc128 func(dst, src, twiddle, scratch []complex128) bool

// wrapCodelet64 adapts a bool-returning kernel to the CodeletFunc signature.
// The bool return is ignored because codelets trust their inputs are pre-validated.
func wrapCodelet64(fn KernelFunc64) CodeletFunc[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) {
		fn(dst, src, twiddle, scratch, bitrev)
	}
}

// wrapCodelet128 adapts a bool-returning kernel to the CodeletFunc signature.
// The bool return is ignored because codelets trust their inputs are pre-validated.
func wrapCodelet128(fn KernelFunc128) CodeletFunc[complex128] {
	return func(dst, src, twiddle, scratch []complex128, bitrev []int) {
		fn(dst, src, twiddle, scratch, bitrev)
	}
}

// wrapCore64 adapts a 4-parameter kernel to the legacy 5-parameter CodeletFunc signature.
func wrapCore64(fn CoreKernelFunc64) CodeletFunc[complex64] {
	return func(dst, src, twiddle, scratch []complex64, _ []int) {
		fn(dst, src, twiddle, scratch)
	}
}

// wrapCore128 adapts a 4-parameter kernel to the legacy 5-parameter CodeletFunc signature.
func wrapCore128(fn CoreKernelFunc128) CodeletFunc[complex128] {
	return func(dst, src, twiddle, scratch []complex128, _ []int) {
		fn(dst, src, twiddle, scratch)
	}
}
