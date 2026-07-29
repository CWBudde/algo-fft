//go:build js && wasm

package main

import (
	"errors"
	"math"
	"math/rand"
	"syscall/js"
	"time"

	algofft "github.com/cwbudde/algo-fft"
)

// Request-boundary limits.
//
// The old bridge clamped n to [16, 4096] and rounded it to a power of two,
// which made every interesting length in the library invisible: no Bluestein,
// no Rader, no mixed-radix. Arbitrary lengths are the point, so the clamp is
// now only what is needed to keep a browser tab alive.
const (
	minAnalyzeN = 2
	maxAnalyzeN = 1 << 20
	minGridSize = 8
	maxGridSize = 256
)

var errUnsupportedPlan = errors.New("unsupported plan type")

// analyzeRequest is the parsed form of the options object.
type analyzeRequest struct {
	n            int
	precision    precisionKind
	strategy     algofft.KernelStrategy
	strategyName string
	planner      algofft.PlannerMode
	freqA        float64
	freqB        float64
	noise        float64
	phase        float64
	delta        float64
	gridSize     int
	grid         bool
	signal       signalKind
	window       windowKind
	roundtrip    bool
}

// parseAnalyzeRequest reads and validates the JS options object.
func parseAnalyzeRequest(opts js.Value) analyzeRequest {
	strategyName := readString(opts, "strategy", algofft.KernelAuto.String())

	return analyzeRequest{
		n:            clampInt(readInt(opts, "n", 1024), minAnalyzeN, maxAnalyzeN),
		precision:    precisionFromString(readString(opts, "precision", precision64.String())),
		strategy:     strategyFromString(strategyName),
		strategyName: strategyName,
		planner:      plannerModeFromString(readString(opts, "planner", "estimate")),
		freqA:        readFloat(opts, "freqA", 6),
		freqB:        readFloat(opts, "freqB", 20),
		noise:        readFloat(opts, "noise", 0.08),
		phase:        readFloat(opts, "phase", 0),
		delta:        readFloat(opts, "delta", defaultCloseDelta),
		gridSize:     clampInt(readInt(opts, "gridSize", 64), minGridSize, maxGridSize),
		grid:         readBool(opts, "grid", true),
		signal:       signalKindFromString(readString(opts, "signal", signalName(sigTone))),
		window:       windowKindFromString(readString(opts, "window", windowName(windowRect))),
		roundtrip:    readBool(opts, "roundtrip", false),
	}
}

// signalParamsOf builds the generator parameters for req, seeded
// deterministically from phase and n exactly as the original inline tone
// generator was.
func (req analyzeRequest) signalParamsOf() signalParams {
	return signalParams{
		freqA: req.freqA,
		freqB: req.freqB,
		noise: req.noise,
		phase: req.phase,
		delta: req.delta,
		seed:  deterministicSeed(req.phase, req.n),
	}
}

// jsAnalyze implements algofft.analyze(). It runs one forward transform of a
// synthetic, windowed signal and returns the waveform, the magnitude and
// phase spectra, an optional 2D magnitude map, and everything the UI needs to
// say which algorithm actually ran.
//
// Shape:
//
//	{
//	  n, gridSize: number,
//	  precision: string,
//	  signal:       Float32Array(n),      // windowed signal actually transformed
//	  spectrum:     Float32Array(n/2),    // magnitude
//	  phase:        Float32Array(n/2),    // radians, (-pi, pi]
//	  gridSpectrum: Float32Array(g*g)|null,   // 2D magnitude
//	  gridPhase:    Float32Array(g*g)|null,   // 2D phase, for domain coloring
//	  window:       Float32Array(n),      // window shape, for the UI overlay
//	  windowName:   string,
//	  coherentGain: number,
//	  reconstruction: Float32Array(n)|undefined,  // only when roundtrip:true
//	  roundtrip: {maxAbsError, rmsError, snrDb}|undefined,
//	  plan:   {algorithm, strategy, strategyRequested, string, planner},
//	  timing: {forwardNs, planNs, cacheHit}
//	}
//
// or {error, panic} on failure.
//
// Callers may pass reusable output buffers as
// out: {signal: {f32, u8}, spectrum: {...}, phase: {...}, gridSpectrum: {...},
//
//	gridPhase: {...}, window: {...}, reconstruction: {...}}
//
// where each pair is a Float32Array and a Uint8Array over the same JS-owned
// ArrayBuffer. Fresh arrays are allocated for any view that is missing or too
// small.
func jsAnalyze(opts js.Value) any {
	if !isObject(opts) {
		opts = js.Global().Get("Object").New()
	}

	req := parseAnalyzeRequest(opts)

	entry, cacheHit, err := planCache.get(planKey{
		kind:      planKind1D,
		precision: req.precision,
		strategy:  req.strategy,
		planner:   req.planner,
		d0:        req.n,
	})
	if err != nil {
		return errorResult(err)
	}

	magCount := req.n / 2
	if magCount < 1 {
		magCount = 1
	}

	bufs := &entry.bufs
	bufs.signal = ensureFloat32(bufs.signal, req.n)
	bufs.mag = ensureFloat32(bufs.mag, magCount)
	bufs.phase = ensureFloat32(bufs.phase, magCount)
	bufs.signal64 = ensureFloat64(bufs.signal64, req.n)
	bufs.windowShape64 = ensureFloat64(bufs.windowShape64, req.n)
	bufs.windowF32 = ensureFloat32(bufs.windowF32, req.n)

	src, dst := complexBuffers(bufs, req.precision, req.n)

	// Generate the raw signal, then apply the window in place BEFORE the
	// transform: bufs.signal64 becomes the actual transform input, and
	// bufs.windowShape64 is the window's own curve for the UI overlay.
	generateSignalKind(bufs.signal64, req.signal, req.signalParamsOf())
	windowShape(bufs.windowShape64, req.window)
	coherentGain := applyWindow(bufs.signal64, req.window)

	float64To32(bufs.signal, bufs.signal64)
	float64To32(bufs.windowF32, bufs.windowShape64)

	// Feed the transform from the float64 signal, not the float32 display
	// copy: rounding to float32 first would throw away complex128's precision
	// advantage before the FFT even runs, and the round-trip check below would
	// then measure float32 rounding instead of the transform's own error.
	fillSourceF64(src, bufs.signal64)

	start := time.Now()

	if err := forwardPlan(entry.plan, dst, src); err != nil {
		return errorResult(err)
	}

	forwardNs := time.Since(start).Nanoseconds()

	fillMagPhase(dst, bufs.mag, bufs.phase)

	var (
		gridArr      any = nil
		gridPhaseArr any = nil
	)

	gridSize := 0

	if req.grid {
		gridMags, gridPhases, err := computeGridSpectrum(req)
		if err != nil {
			return errorResult(err)
		}

		gridSize = req.gridSize
		gridArr = writeFloat32(outView(opts, "gridSpectrum"), gridMags)
		gridPhaseArr = writeFloat32(outView(opts, "gridPhase"), gridPhases)
	}

	algorithm, resolved := describePlan(entry)

	result := map[string]any{
		"n":            req.n,
		"precision":    req.precision.String(),
		"signal":       writeFloat32(outView(opts, "signal"), bufs.signal),
		"spectrum":     writeFloat32(outView(opts, "spectrum"), bufs.mag),
		"phase":        writeFloat32(outView(opts, "phase"), bufs.phase),
		"gridSpectrum": gridArr,
		"gridPhase":    gridPhaseArr,
		"gridSize":     gridSize,
		"window":       writeFloat32(outView(opts, "window"), bufs.windowF32),
		"windowName":   windowName(req.window),
		"coherentGain": coherentGain,
		"plan": map[string]any{
			"algorithm": algorithm,
			// Requested and resolved are always reported separately. A forced
			// strategy the planner cannot honour is silently downgraded, and
			// two identical-looking curves would otherwise invite exactly the
			// wrong conclusion. Note also that Rader is not a strategy: a Rader
			// plan reports Bluestein here and only names "rader" in algorithm.
			"strategy":          resolved,
			"strategyRequested": req.strategy.String(),
			"planner":           plannerModeName(req.planner),
			"string":            entry.info.String(),
		},
		"timing": map[string]any{
			"forwardNs": float64(forwardNs),
			"planNs":    float64(entry.buildNs),
			"cacheHit":  cacheHit,
		},
	}

	if req.roundtrip {
		reconDst := complexReconBuffer(bufs, req.precision, req.n)
		if err := inversePlan(entry.plan, reconDst, dst); err != nil {
			return errorResult(err)
		}

		bufs.reconReal = ensureFloat64(bufs.reconReal, req.n)
		bufs.reconF32 = ensureFloat32(bufs.reconF32, req.n)

		extractReal(reconDst, bufs.reconReal)
		float64To32(bufs.reconF32, bufs.reconReal)

		// Inverse() is already 1/N-normalized; do not divide again. The error
		// is measured against the windowed float64 signal (bufs.signal64), not
		// the float32 display copy — comparing against float32 would mask the
		// ~1e-15 vs ~1e-6 precision gap between complex128 and complex64.
		maxAbsErr, rmsErr, snrDb := computeRoundtripMetrics(bufs.signal64, bufs.reconReal)

		result["reconstruction"] = writeFloat32(outView(opts, "reconstruction"), bufs.reconF32)
		result["roundtrip"] = map[string]any{
			"maxAbsError": maxAbsErr,
			"rmsError":    rmsErr,
			"snrDb":       snrDb,
		}
	}

	return js.ValueOf(result)
}

// jsCacheStats implements algofft.cacheStats().
func jsCacheStats(_ js.Value) any {
	return js.ValueOf(planCache.stats())
}

// jsCacheClear implements algofft.cacheClear(), returning the number of plans
// dropped.
func jsCacheClear(_ js.Value) any {
	return js.ValueOf(map[string]any{"dropped": planCache.clear()})
}

// computeGridSpectrum runs the 2D transform used by the frequency map panel
// and returns its magnitudes.
// computeGridSpectrum returns the magnitude and phase of the 2D transform. The
// phase is what lets the frequency map be domain-colored (hue from phase,
// lightness from log magnitude) rather than colored from magnitude alone.
func computeGridSpectrum(req analyzeRequest) (mags, phases []float32, err error) {
	entry, _, err := planCache.get(planKey{
		kind:      planKind2D,
		precision: req.precision,
		strategy:  req.strategy,
		planner:   req.planner,
		d0:        req.gridSize,
		d1:        req.gridSize,
	})
	if err != nil {
		return nil, nil, err
	}

	size := req.gridSize
	count := size * size

	bufs := &entry.bufs
	bufs.signal = ensureFloat32(bufs.signal, count)
	bufs.mag = ensureFloat32(bufs.mag, count)
	bufs.phase = ensureFloat32(bufs.phase, count)

	src, dst := complexBuffers(bufs, req.precision, count)

	generateGrid(bufs.signal, size, req)
	fillSource(src, bufs.signal)

	if err := forward2DPlan(entry.plan, dst, src); err != nil {
		return nil, nil, err
	}

	fillMagPhase(dst, bufs.mag, bufs.phase)

	return bufs.mag, bufs.phase, nil
}

// generateGrid fills dst (size*size, row-major) with the demo's synthetic 2D
// pattern.
func generateGrid(dst []float32, size int, req analyzeRequest) {
	rng := rand.New(rand.NewSource(int64(math.Round(req.phase*830)) + int64(size)*97)) //nolint:gosec // deterministic demo noise

	for y := 0; y < size; y++ {
		fy := float64(y) / float64(size)

		for x := 0; x < size; x++ {
			fx := float64(x) / float64(size)
			val := math.Sin(2*math.Pi*req.freqA*fx+req.phase*0.6) +
				0.8*math.Sin(2*math.Pi*req.freqB*fy+req.phase*0.4) +
				0.45*math.Sin(2*math.Pi*(req.freqA*0.5*fx+req.freqB*0.5*fy)+req.phase*0.2)

			if req.noise > 0 {
				val += (rng.Float64()*2 - 1) * req.noise
			}

			dst[y*size+x] = float32(val)
		}
	}
}

// complexBuffers returns the cached src/dst slices for the requested precision,
// growing them if needed. The returned values are []complex64 or []complex128
// boxed in an any; the concrete type is recovered by one type switch per call
// site rather than one per element.
func complexBuffers(bufs *planBuffers, precision precisionKind, n int) (src, dst any) {
	if precision == precision128 {
		bufs.src128 = ensureComplex128(bufs.src128, n)
		bufs.dst128 = ensureComplex128(bufs.dst128, n)

		return bufs.src128, bufs.dst128
	}

	bufs.src64 = ensureComplex64(bufs.src64, n)
	bufs.dst64 = ensureComplex64(bufs.dst64, n)

	return bufs.src64, bufs.dst64
}

// complexReconBuffer returns the cached reconstruction slice for the
// requested precision, growing it if needed. Kept separate from
// complexBuffers because src/dst and the round-trip reconstruction have
// independent lifetimes: a caller may re-run the forward transform without
// ever asking for a round-trip.
func complexReconBuffer(bufs *planBuffers, precision precisionKind, n int) any {
	if precision == precision128 {
		bufs.reconC128 = ensureComplex128(bufs.reconC128, n)

		return bufs.reconC128
	}

	bufs.reconC64 = ensureComplex64(bufs.reconC64, n)

	return bufs.reconC64
}

// float64To32 downcasts src into dst element-wise. dst must be at least as
// long as src.
func float64To32(dst []float32, src []float64) {
	for i, v := range src {
		dst[i] = float32(v)
	}
}

// extractReal writes the real part of a complex buffer into out.
func extractReal(src any, out []float64) {
	switch s := src.(type) {
	case []complex64:
		for i := range out {
			out[i] = float64(real(s[i]))
		}
	case []complex128:
		for i := range out {
			out[i] = real(s[i])
		}
	}
}

// computeRoundtripMetrics compares reconstruction against original,
// element-wise, both in float64 so the comparison itself never becomes the
// precision bottleneck.
func computeRoundtripMetrics(original, reconstruction []float64) (maxAbsErr, rmsErr, snrDb float64) {
	n := len(original)
	if n == 0 || len(reconstruction) < n {
		return 0, 0, 0
	}

	var sumSq, sumErrSq float64

	for i := 0; i < n; i++ {
		e := reconstruction[i] - original[i]
		if ae := math.Abs(e); ae > maxAbsErr {
			maxAbsErr = ae
		}

		sumErrSq += e * e
		sumSq += original[i] * original[i]
	}

	rmsErr = math.Sqrt(sumErrSq / float64(n))

	switch {
	case sumErrSq == 0:
		snrDb = math.Inf(1)
	case sumSq == 0:
		snrDb = 0
	default:
		snrDb = 10 * math.Log10(sumSq/sumErrSq)
	}

	return maxAbsErr, rmsErr, snrDb
}

// fillSource copies a real-valued signal into a complex source buffer.
func fillSource(src any, signal []float32) {
	switch s := src.(type) {
	case []complex64:
		for i, v := range signal {
			s[i] = complex(v, 0)
		}
	case []complex128:
		for i, v := range signal {
			s[i] = complex(float64(v), 0)
		}
	}
}

// fillSourceF64 copies a real-valued, full-precision signal into a complex
// source buffer. Unlike fillSource, this never routes through a float32
// intermediate, so a complex128 destination keeps the input's full precision
// — required for the round-trip check to measure the transform's own error
// rather than a prior float32 rounding step.
func fillSourceF64(src any, signal []float64) {
	switch s := src.(type) {
	case []complex64:
		for i, v := range signal {
			s[i] = complex(float32(v), 0)
		}
	case []complex128:
		for i, v := range signal {
			s[i] = complex(v, 0)
		}
	}
}

// fillMagPhase extracts magnitude and phase for the first len(mag) bins.
func fillMagPhase(dst any, mag, phase []float32) {
	switch d := dst.(type) {
	case []complex64:
		for i := range mag {
			re, im := float64(real(d[i])), float64(imag(d[i]))
			mag[i] = float32(math.Hypot(re, im))
			phase[i] = float32(math.Atan2(im, re))
		}
	case []complex128:
		for i := range mag {
			re, im := real(d[i]), imag(d[i])
			mag[i] = float32(math.Hypot(re, im))
			phase[i] = float32(math.Atan2(im, re))
		}
	}
}

// forwardPlan runs a 1D forward transform on whichever precision the cached
// plan was built for.
func forwardPlan(plan, dst, src any) error {
	switch p := plan.(type) {
	case *algofft.Plan[complex64]:
		return p.Forward(dst.([]complex64), src.([]complex64))
	case *algofft.Plan[complex128]:
		return p.Forward(dst.([]complex128), src.([]complex128))
	default:
		return errUnsupportedPlan
	}
}

// inversePlan runs a 1D inverse transform on whichever precision the cached
// plan was built for. Plan.Inverse is already 1/N-normalized.
func inversePlan(plan, dst, src any) error {
	switch p := plan.(type) {
	case *algofft.Plan[complex64]:
		return p.Inverse(dst.([]complex64), src.([]complex64))
	case *algofft.Plan[complex128]:
		return p.Inverse(dst.([]complex128), src.([]complex128))
	default:
		return errUnsupportedPlan
	}
}

// forward2DPlan runs a 2D forward transform on whichever precision the cached
// plan was built for.
func forward2DPlan(plan, dst, src any) error {
	switch p := plan.(type) {
	case *algofft.Plan2D[complex64]:
		return p.Forward(dst.([]complex64), src.([]complex64))
	case *algofft.Plan2D[complex128]:
		return p.Forward(dst.([]complex128), src.([]complex128))
	default:
		return errUnsupportedPlan
	}
}
