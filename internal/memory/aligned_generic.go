package memory

import "github.com/cwbudde/algo-fft/internal/fftypes"

// AllocAligned allocates a SIMD-aligned slice of n elements of the complex type T
// together with its []byte backing, which must be kept alive to prevent the GC
// from reclaiming the aligned memory. It dispatches to AllocAlignedComplex64 or
// AllocAlignedComplex128; the default branch is unreachable under the Complex
// constraint and exists only to keep the generic type-switch total.
func AllocAligned[T fftypes.Complex](n int) ([]T, []byte) {
	var zero T

	switch any(zero).(type) {
	case complex64:
		s, backing := AllocAlignedComplex64(n)
		return any(s).([]T), backing
	case complex128:
		s, backing := AllocAlignedComplex128(n)
		return any(s).([]T), backing
	default:
		return make([]T, n), nil
	}
}
