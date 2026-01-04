package fft

import "github.com/MeKo-Christian/algo-fft/internal/math"

var (
	bitrevSize8Mixed24    = math.ComputeBitReversalIndicesMixed24(8)
	bitrevSize32Mixed24   = math.ComputeBitReversalIndicesMixed24(32)
	bitrevSize128Mixed24  = math.ComputeBitReversalIndicesMixed24(128)
	bitrevSize512Mixed24  = math.ComputeBitReversalIndicesMixed24(512)
	bitrevSize2048Mixed24 = math.ComputeBitReversalIndicesMixed24(2048)
	bitrevSize8192Mixed24 = math.ComputeBitReversalIndicesMixed24(8192)
)
