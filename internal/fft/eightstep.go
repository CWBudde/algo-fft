package fft

import "github.com/MeKo-Christian/algo-fft/internal/kernels"

// Re-export eight-step FFT functions from internal/kernels for backward compatibility.

var (
	forwardEightStepComplex64  = kernels.ForwardEightStepComplex64
	inverseEightStepComplex64  = kernels.InverseEightStepComplex64
	forwardEightStepComplex128 = kernels.ForwardEightStepComplex128
	inverseEightStepComplex128 = kernels.InverseEightStepComplex128
)
