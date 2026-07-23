package kernels

//go:generate go run ../../cmd/gencodelets .
//go:generate go run ../../cmd/gencodelets -inventory ../../docs/IMPLEMENTATION_INVENTORY.md

import (
	"github.com/cwbudde/algo-fft/internal/planner"
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
	SIMDSSE3   = planner.SIMDSSE3
	SIMDAVX2   = planner.SIMDAVX2
	SIMDAVX512 = planner.SIMDAVX512
	SIMDNEON   = planner.SIMDNEON
)

// KernelType constants for codelet classification.
const (
	KernelTypeCore = planner.KernelTypeCore
	KernelTypeDIT  = planner.KernelTypeDIT
)

// GetRegistry returns the appropriate registry for type T.
func GetRegistry[T Complex]() *planner.CodeletRegistry[T] {
	return planner.GetRegistry[T]()
}

// This file registers all built-in codelets with the global registries.
// Registration happens at init time so codelets are available when plans are created.
// The register*DITCodelets* functions are generated from cmd/gencodelets/specs.go;
// run `go generate ./internal/kernels/...` after editing the spec table.

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

	// Register AVX-512 codelets (conditional on build tags)
	registerAVX512DITCodelets64()
	registerAVX512DITCodelets128()
}
