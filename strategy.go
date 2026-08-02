package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
)

// KernelStrategy controls which FFT kernel a plan should use. It is declared
// in this package (not aliased from an internal package) so internal
// refactors cannot change the public API; conversions happen at the
// plan-construction boundary.
type KernelStrategy uint8

// The constants below use explicit numeric values (not iota) so that the
// removal of KernelEightStep (value 4) does not silently renumber
// KernelBluestein and everything after it. See strategy_numbering_test.go.
const (
	KernelAuto     KernelStrategy = 0 // Let the planner choose by size
	KernelDIT      KernelStrategy = 1 // Decimation-in-time
	KernelStockham KernelStrategy = 2 // Stockham autosort
	KernelSixStep  KernelStrategy = 3 // Six-step (cache-friendly, large sizes)
	// 4 was KernelEightStep, removed 2026-08-02 as a duplicate implementation
	// of KernelSixStep (same algorithm, different name). The value is
	// retired and must not be reused.
	KernelBluestein  KernelStrategy = 5 // Bluestein (arbitrary lengths)
	KernelRecursive  KernelStrategy = 6 // Recursive decomposition with codelet leaves
	KernelSplitRadix KernelStrategy = 7 // Split-radix (2/4) DIT (power-of-two lengths)
	KernelFourStep   KernelStrategy = 8 // Four-step (cache-blocked rectangular six-step, power-of-two lengths)
	// KernelMixedRadix is the mixed-radix engine (factors 2/3/5/7/11). Plans
	// report it for every non-power-of-two length that does not run Bluestein;
	// forcing it at a length it is not the route for falls back to the size
	// heuristic, since the reported strategy always names the executed route.
	KernelMixedRadix KernelStrategy = 9
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
	case KernelBluestein:
		return strategyNameBluestein
	case KernelRecursive:
		return strategyNameRecursive
	case KernelSplitRadix:
		return strategyNameSplitRadix
	case KernelFourStep:
		return strategyNameFourStep
	case KernelMixedRadix:
		return strategyNameMixedRadix
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
	case KernelBluestein:
		return fftypes.KernelBluestein
	case KernelRecursive:
		return fftypes.KernelRecursive
	case KernelSplitRadix:
		return fftypes.KernelSplitRadix
	case KernelFourStep:
		return fftypes.KernelFourStep
	case KernelMixedRadix:
		return fftypes.KernelMixedRadix
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
	case fftypes.KernelBluestein:
		return KernelBluestein
	case fftypes.KernelRecursive:
		return KernelRecursive
	case fftypes.KernelSplitRadix:
		return KernelSplitRadix
	case fftypes.KernelFourStep:
		return KernelFourStep
	case fftypes.KernelMixedRadix:
		return KernelMixedRadix
	case fftypes.KernelAuto:
		return KernelAuto
	default:
		return KernelAuto
	}
}
