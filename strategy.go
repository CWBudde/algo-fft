package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
)

// KernelStrategy controls which FFT kernel a plan should use. It is declared
// in this package (not aliased from an internal package) so internal
// refactors cannot change the public API; conversions happen at the
// plan-construction boundary.
type KernelStrategy uint8

const (
	KernelAuto       KernelStrategy = iota // Let the planner choose by size
	KernelDIT                              // Decimation-in-time
	KernelStockham                         // Stockham autosort
	KernelSixStep                          // Six-step (cache-friendly, large sizes)
	KernelEightStep                        // Eight-step (cache-friendly, large sizes)
	KernelBluestein                        // Bluestein (arbitrary lengths)
	KernelRecursive                        // Recursive decomposition with codelet leaves
	KernelSplitRadix                       // Split-radix (2/4) DIT (power-of-two lengths)
)

// String returns a human-readable name for the strategy.
func (s KernelStrategy) String() string {
	switch s {
	case KernelAuto:
		return strategyNameAuto
	case KernelDIT:
		return strategyNameDIT
	case KernelStockham:
		return strategyNameStockham
	case KernelSixStep:
		return strategyNameSixStep
	case KernelEightStep:
		return strategyNameEightStep
	case KernelBluestein:
		return strategyNameBluestein
	case KernelRecursive:
		return strategyNameRecursive
	case KernelSplitRadix:
		return strategyNameSplitRadix
	default:
		return "unknown"
	}
}

// internal converts the public strategy to the internal kernel-strategy enum.
func (s KernelStrategy) internal() fftypes.KernelStrategy {
	switch s {
	case KernelDIT:
		return fftypes.KernelDIT
	case KernelStockham:
		return fftypes.KernelStockham
	case KernelSixStep:
		return fftypes.KernelSixStep
	case KernelEightStep:
		return fftypes.KernelEightStep
	case KernelBluestein:
		return fftypes.KernelBluestein
	case KernelRecursive:
		return fftypes.KernelRecursive
	case KernelSplitRadix:
		return fftypes.KernelSplitRadix
	case KernelAuto:
		return fftypes.KernelAuto
	default:
		return fftypes.KernelAuto
	}
}

// kernelStrategyFromInternal converts an internal kernel-strategy value to
// the public enum.
func kernelStrategyFromInternal(s fftypes.KernelStrategy) KernelStrategy {
	switch s {
	case fftypes.KernelDIT:
		return KernelDIT
	case fftypes.KernelStockham:
		return KernelStockham
	case fftypes.KernelSixStep:
		return KernelSixStep
	case fftypes.KernelEightStep:
		return KernelEightStep
	case fftypes.KernelBluestein:
		return KernelBluestein
	case fftypes.KernelRecursive:
		return KernelRecursive
	case fftypes.KernelSplitRadix:
		return KernelSplitRadix
	case fftypes.KernelAuto:
		return KernelAuto
	default:
		return KernelAuto
	}
}
