//go:build js && wasm

package main

import (
	"runtime"
	"runtime/debug"
	"sync"
	"syscall/js"
	"time"

	algofft "github.com/cwbudde/algo-fft"
)

// maxStrategyProbe bounds the enum walk below. It only needs to exceed the
// number of kernel strategies the library will ever have.
const maxStrategyProbe = 64

// strategyNames returns every KernelStrategy the linked library knows about,
// in declaration order.
//
// The list is derived, never written out by hand. The enum has grown twice
// already — KernelFourStep and then KernelMixedRadix — and a hardcoded list
// silently drifts out of date, which in a demo means offering a strategy the
// library dropped or hiding one it gained. Walking the values until String()
// stops recognising them keeps the UI honest for free.
func strategyNames() []string {
	names := make([]string, 0, 12)

	for i := 0; i < maxStrategyProbe; i++ {
		name := algofft.KernelStrategy(i).String() //nolint:gosec // bounded by maxStrategyProbe
		if name == "unknown" {
			break
		}

		names = append(names, name)
	}

	return names
}

// strategyFromString resolves a strategy name (as produced by strategyNames,
// case-sensitive) to its enum value. Unknown names resolve to KernelAuto.
func strategyFromString(name string) algofft.KernelStrategy {
	for i := 0; i < maxStrategyProbe; i++ {
		s := algofft.KernelStrategy(i) //nolint:gosec // bounded by maxStrategyProbe

		candidate := s.String()
		if candidate == "unknown" {
			break
		}

		if candidate == name {
			return s
		}
	}

	return algofft.KernelAuto
}

// plannerModeName names a planner mode. The library exports no String method
// for PlannerMode, so this table is the demo's own; the enum walk in
// plannerModeNames is driven by it and stops at the first unrecognised value.
func plannerModeName(m algofft.PlannerMode) string {
	switch m {
	case algofft.PlannerEstimate:
		return "estimate"
	case algofft.PlannerMeasure:
		return "measure"
	case algofft.PlannerPatient:
		return "patient"
	case algofft.PlannerExhaustive:
		return "exhaustive"
	default:
		return "unknown"
	}
}

// plannerModeNames returns every planner mode in declaration order.
func plannerModeNames() []string {
	names := make([]string, 0, 4)

	for i := 0; i < maxStrategyProbe; i++ {
		name := plannerModeName(algofft.PlannerMode(i)) //nolint:gosec // bounded by maxStrategyProbe
		if name == "unknown" {
			break
		}

		names = append(names, name)
	}

	return names
}

// plannerModeFromString resolves a planner mode name to its enum value,
// defaulting to PlannerEstimate.
func plannerModeFromString(name string) algofft.PlannerMode {
	for i := 0; i < maxStrategyProbe; i++ {
		m := algofft.PlannerMode(i) //nolint:gosec // bounded by maxStrategyProbe

		candidate := plannerModeName(m)
		if candidate == "unknown" {
			break
		}

		if candidate == name {
			return m
		}
	}

	return algofft.PlannerEstimate
}

var (
	timerGranularityOnce sync.Once
	timerGranularity     time.Duration
)

// measureTimerGranularity probes the smallest interval time.Now() can actually
// resolve.
//
// Under js/wasm time.Now() is backed by performance.now(), which browsers clamp
// as a Spectre mitigation: 100 microseconds in Chrome, around 1 millisecond in
// Firefox and Safari, unless the page is cross-origin isolated (GitHub Pages
// cannot send the COOP/COEP headers that would make it so). A single transform
// below roughly n=2^16 therefore measures as either zero or one tick. Any
// timing the demo shows has to be interpreted against this number, so it is
// probed rather than assumed.
//
// The method: spin on time.Now() until the reading changes, 32 times, and take
// the smallest non-zero delta observed.
func measureTimerGranularity() time.Duration {
	const (
		samples  = 32
		spinCap  = 1 << 22 // guards against a pathologically stuck clock
		fallback = time.Millisecond
	)

	best := time.Duration(0)

	for i := 0; i < samples; i++ {
		start := time.Now()
		delta := time.Duration(0)

		for spin := 0; spin < spinCap; spin++ {
			delta = time.Since(start)
			if delta > 0 {
				break
			}
		}

		if delta <= 0 {
			continue
		}

		if best == 0 || delta < best {
			best = delta
		}
	}

	if best == 0 {
		return fallback
	}

	return best
}

// timerGranularityNs returns the probed timer granularity in nanoseconds,
// measuring once on first use.
func timerGranularityNs() float64 {
	timerGranularityOnce.Do(func() {
		timerGranularity = measureTimerGranularity()
	})

	return float64(timerGranularity.Nanoseconds())
}

// buildVersion reports the module version this wasm binary was built from.
func buildVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}

	if info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}

	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" && setting.Value != "" {
			if len(setting.Value) > 12 {
				return "devel+" + setting.Value[:12]
			}

			return "devel+" + setting.Value
		}
	}

	return "devel"
}

// jsInfo implements algofft.info(). It takes no arguments.
//
// Shape:
//
//	{
//	  version, goVersion, goos, goarch: string,
//	  simd: false,
//	  strategies: string[],      // KernelStrategy names, index == enum value
//	  plannerModes: string[],    // PlannerMode names, index == enum value
//	  precisions: string[],      // ["complex64", "complex128"]
//	  timerGranularityNs: number,
//	  minN, maxN, planCacheCapacity: number
//	}
func jsInfo(_ js.Value) any {
	strategies := strategyNames()
	strategyVals := make([]any, len(strategies))

	for i, name := range strategies {
		strategyVals[i] = name
	}

	modes := plannerModeNames()
	modeVals := make([]any, len(modes))

	for i, name := range modes {
		modeVals[i] = name
	}

	return js.ValueOf(map[string]any{
		"version":   buildVersion(),
		"goVersion": runtime.Version(),
		"goos":      runtime.GOOS,
		"goarch":    runtime.GOARCH,
		// The wasm build has no SIMD path: the library's vector kernels are
		// amd64/arm64 assembly and there is no wasm128 backend. Reported so the
		// benchmark page can say so instead of implying peak numbers.
		"simd":               false,
		"strategies":         strategyVals,
		"plannerModes":       modeVals,
		"precisions":         []any{precision64.String(), precision128.String()},
		"timerGranularityNs": timerGranularityNs(),
		"minN":               minAnalyzeN,
		"maxN":               maxAnalyzeN,
		"planCacheCapacity":  planCacheCapacity,
	})
}
