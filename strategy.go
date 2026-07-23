package algofft

import "github.com/cwbudde/algo-fft/internal/fft"

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
func (s KernelStrategy) internal() fft.KernelStrategy {
	switch s {
	case KernelDIT:
		return fft.KernelDIT
	case KernelStockham:
		return fft.KernelStockham
	case KernelSixStep:
		return fft.KernelSixStep
	case KernelEightStep:
		return fft.KernelEightStep
	case KernelBluestein:
		return fft.KernelBluestein
	case KernelRecursive:
		return fft.KernelRecursive
	case KernelSplitRadix:
		return fft.KernelSplitRadix
	case KernelAuto:
		return fft.KernelAuto
	default:
		return fft.KernelAuto
	}
}

// kernelStrategyFromInternal converts an internal kernel-strategy value to
// the public enum.
func kernelStrategyFromInternal(s fft.KernelStrategy) KernelStrategy {
	switch s {
	case fft.KernelDIT:
		return KernelDIT
	case fft.KernelStockham:
		return KernelStockham
	case fft.KernelSixStep:
		return KernelSixStep
	case fft.KernelEightStep:
		return KernelEightStep
	case fft.KernelBluestein:
		return KernelBluestein
	case fft.KernelRecursive:
		return KernelRecursive
	case fft.KernelSplitRadix:
		return KernelSplitRadix
	case fft.KernelAuto:
		return KernelAuto
	default:
		return KernelAuto
	}
}
