package kernels

import (
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// Re-export planner types for backward compatibility

// CodeletFunc is a type alias from the planner package.
type CodeletFunc[T Complex] = planner.CodeletFunc[T]

// SIMDLevel is a type alias from the planner package.
type SIMDLevel = planner.SIMDLevel

// SIMD level constants
const (
	SIMDNone   = planner.SIMDNone
	SIMDSSE2   = planner.SIMDSSE2
	SIMDAVX2   = planner.SIMDAVX2
	SIMDAVX512 = planner.SIMDAVX512
	SIMDNEON   = planner.SIMDNEON
)

// BitrevFunc is a type alias from the planner package.
type BitrevFunc = planner.BitrevFunc

// CodeletEntry is a type alias from the planner package.
type CodeletEntry[T Complex] = planner.CodeletEntry[T]

// CodeletRegistry is a type alias from the planner package.
type CodeletRegistry[T Complex] = planner.CodeletRegistry[T]

// NewCodeletRegistry creates a new codelet registry.
func NewCodeletRegistry[T Complex]() *CodeletRegistry[T] {
	return planner.NewCodeletRegistry[T]()
}

// GetRegistry returns the appropriate codelet registry for type T.
func GetRegistry[T Complex]() *CodeletRegistry[T] {
	return planner.GetRegistry[T]()
}

// Global registries
var (
	Registry64  = planner.Registry64
	Registry128 = planner.Registry128
)
