package planner

import "github.com/cwbudde/algo-fft/internal/fftypes"

// Complex is a type alias for the complex number constraint.
// The canonical definition is in internal/fftypes.
type Complex = fftypes.Complex

// KernelStrategy is a type alias for the kernel strategy enum.
// The canonical definition is in internal/fftypes.
type KernelStrategy = fftypes.KernelStrategy

// Strategy constants.
const (
	KernelAuto      = fftypes.KernelAuto
	KernelDIT       = fftypes.KernelDIT
	KernelStockham  = fftypes.KernelStockham
	KernelSixStep   = fftypes.KernelSixStep
	KernelEightStep = fftypes.KernelEightStep
	KernelBluestein = fftypes.KernelBluestein
	KernelRecursive = fftypes.KernelRecursive
)
