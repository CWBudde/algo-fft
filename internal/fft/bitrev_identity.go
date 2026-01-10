package fft

import "github.com/MeKo-Christian/algo-fft/internal/math"

var (
	bitrevSize16Identity = math.ComputeIdentityIndices(16)
	bitrevSize32Identity = math.ComputeIdentityIndices(32)
	bitrevSize64Radix2   = math.ComputeBitReversalIndices(64)
)
