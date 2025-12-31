package fft

import "github.com/MeKo-Christian/algo-fft/internal/kernels"

// Re-export six-step FFT functions from internal/kernels for backward compatibility.

var (
	forwardSixStepComplex64  = kernels.ForwardSixStepComplex64
	inverseSixStepComplex64  = kernels.InverseSixStepComplex64
	forwardSixStepComplex128 = kernels.ForwardSixStepComplex128
	inverseSixStepComplex128 = kernels.InverseSixStepComplex128
)
