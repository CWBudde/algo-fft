//go:build js && wasm

package main

import (
	"syscall/js"
	"time"

	algofft "github.com/cwbudde/algo-fft"
)

// roundtripCompareRequest is the parsed form of roundtripCompare's options.
type roundtripCompareRequest struct {
	n        int
	strategy algofft.KernelStrategy
	planner  algofft.PlannerMode
	signal   signalKind
	window   windowKind
	freqA    float64
	freqB    float64
	noise    float64
	phase    float64
	delta    float64
}

func parseRoundtripCompareRequest(opts js.Value) roundtripCompareRequest {
	return roundtripCompareRequest{
		n:        clampInt(readInt(opts, "n", 1024), minAnalyzeN, maxAnalyzeN),
		strategy: strategyFromString(readString(opts, "strategy", algofft.KernelAuto.String())),
		planner:  plannerModeFromString(readString(opts, "planner", "estimate")),
		signal:   signalKindFromString(readString(opts, "signal", signalName(sigTone))),
		window:   windowKindFromString(readString(opts, "window", windowName(windowRect))),
		freqA:    readFloat(opts, "freqA", 6),
		freqB:    readFloat(opts, "freqB", 20),
		noise:    readFloat(opts, "noise", 0.08),
		phase:    readFloat(opts, "phase", 0),
		delta:    readFloat(opts, "delta", defaultCloseDelta),
	}
}

// jsRoundtripCompare implements algofft.roundtripCompare(). It runs the same
// windowed signal forward-then-inverse at both complex64 and complex128 and
// reports the round-trip error at each precision, side by side. The ~1e-6 vs
// ~1e-15 gap is the clearest available demonstration of what the extra
// precision buys.
//
// Shape:
//
//	{
//	  c64:  {maxAbsError, rmsError, snrDb, forwardNs},
//	  c128: {maxAbsError, rmsError, snrDb, forwardNs},
//	}
//
// or {error, panic} on failure.
func jsRoundtripCompare(opts js.Value) any {
	if !isObject(opts) {
		opts = js.Global().Get("Object").New()
	}

	req := parseRoundtripCompareRequest(opts)

	raw := make([]float64, req.n)
	generateSignalKind(raw, req.signal, signalParams{
		freqA: req.freqA,
		freqB: req.freqB,
		noise: req.noise,
		phase: req.phase,
		delta: req.delta,
		seed:  deterministicSeed(req.phase, req.n),
	})
	applyWindow(raw, req.window)

	c64, err := roundtripAt(precision64, req.n, req.strategy, req.planner, raw)
	if err != nil {
		return errorResult(err)
	}

	c128, err := roundtripAt(precision128, req.n, req.strategy, req.planner, raw)
	if err != nil {
		return errorResult(err)
	}

	return js.ValueOf(map[string]any{
		"c64":  c64,
		"c128": c128,
	})
}

// roundtripAt builds (or reuses) a plan at the given precision, runs
// forward+inverse on raw, and returns the round-trip metrics plus the
// forward-transform time.
func roundtripAt(
	precision precisionKind,
	n int,
	strategy algofft.KernelStrategy,
	planner algofft.PlannerMode,
	raw []float64,
) (map[string]any, error) {
	entry, _, err := planCache.get(planKey{
		kind:      planKind1D,
		precision: precision,
		strategy:  strategy,
		planner:   planner,
		d0:        n,
	})
	if err != nil {
		return nil, err
	}

	var (
		reconReal []float64
		forwardNs int64
	)

	if precision == precision128 {
		src := make([]complex128, n)
		dst := make([]complex128, n)
		recon := make([]complex128, n)

		for i, v := range raw {
			src[i] = complex(v, 0)
		}

		start := time.Now()

		if err := forwardPlan(entry.plan, dst, src); err != nil {
			return nil, err
		}

		forwardNs = time.Since(start).Nanoseconds()

		if err := inversePlan(entry.plan, recon, dst); err != nil {
			return nil, err
		}

		reconReal = make([]float64, n)
		for i := range recon {
			reconReal[i] = real(recon[i])
		}
	} else {
		src := make([]complex64, n)
		dst := make([]complex64, n)
		recon := make([]complex64, n)

		for i, v := range raw {
			src[i] = complex(float32(v), 0)
		}

		start := time.Now()

		if err := forwardPlan(entry.plan, dst, src); err != nil {
			return nil, err
		}

		forwardNs = time.Since(start).Nanoseconds()

		if err := inversePlan(entry.plan, recon, dst); err != nil {
			return nil, err
		}

		reconReal = make([]float64, n)
		for i := range recon {
			reconReal[i] = float64(real(recon[i]))
		}
	}

	maxAbsErr, rmsErr, snrDb := computeRoundtripMetrics(raw, reconReal)

	return map[string]any{
		"maxAbsError": maxAbsErr,
		"rmsError":    rmsErr,
		"snrDb":       snrDb,
		"forwardNs":   float64(forwardNs),
	}, nil
}
