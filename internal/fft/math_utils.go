package fft

import m "github.com/MeKo-Christian/algo-fft/internal/math"

// Re-export public functions from internal/math
var (
	IsPowerOf2     = m.IsPowerOf2
	NextPowerOfTwo = m.NextPowerOfTwo
)

// Private re-exports for internal use
var (
	isPowerOf  = m.IsPowerOf
	isPowerOf3 = m.IsPowerOf3
	isPowerOf4 = m.IsPowerOf4
	isPowerOf5 = m.IsPowerOf5
)
