package fft

import "github.com/MeKo-Christian/algo-fft/internal/math"

var (
	bitrevSize16Radix4  = math.ComputeBitReversalIndicesRadix4(16)
	bitrevSize64Radix4  = math.ComputeBitReversalIndicesRadix4(64)
	bitrevSize128Radix4 = math.ComputeBitReversalIndicesRadix4(128)
	bitrevSize256Radix4 = math.ComputeBitReversalIndicesRadix4(256)
)
