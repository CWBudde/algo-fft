package fft

import m "github.com/MeKo-Christian/algo-fft/internal/math"

// Re-export from internal/math
const AlignmentBytes = m.AlignmentBytes

var (
	AllocAlignedComplex64 = m.AllocAlignedComplex64
	AllocAlignedComplex128 = m.AllocAlignedComplex128
	alignPtr              = m.AlignPtr
)
