//go:build js && wasm

package main

import (
	"math"
	"syscall/js"
	"time"
)

// maxConvolveN and maxConvolveKernel bound the naive O(N*M) reference
// computed alongside the FFT-based result, per the "cap at ~4096x256"
// requirement — large enough to show the crossover, small enough that the
// direct reference never stalls the tab.
const (
	maxConvolveN      = 4096
	maxConvolveKernel = 256
)

// convolveRequest is the parsed form of convolve's options object.
type convolveRequest struct {
	n          int
	signal     signalKind
	window     windowKind
	freqA      float64
	freqB      float64
	noise      float64
	phase      float64
	delta      float64
	kernelName string
	correlate  bool
}

func parseConvolveRequest(opts js.Value) convolveRequest {
	return convolveRequest{
		n:          clampInt(readInt(opts, "n", 512), minAnalyzeN, maxConvolveN),
		signal:     signalKindFromString(readString(opts, "signal", signalName(sigTone))),
		window:     windowKindFromString(readString(opts, "window", windowName(windowRect))),
		freqA:      readFloat(opts, "freqA", 6),
		freqB:      readFloat(opts, "freqB", 20),
		noise:      readFloat(opts, "noise", 0.08),
		phase:      readFloat(opts, "phase", 0),
		delta:      readFloat(opts, "delta", defaultCloseDelta),
		kernelName: readString(opts, "kernel", "lowpass"),
		correlate:  readBool(opts, "correlate", false),
	}
}

// jsConvolve implements algofft.convolve(). It builds signal `a` from the
// current signal generator, a small kernel `b` from the named preset, and
// runs them through the demo's cached, persistent Convolver/Correlator
// (never the one-shot Convolve/Correlate free functions), alongside a naive
// O(N*M) direct reference so the FFT-vs-direct crossover is visible.
//
// Shape:
//
//	{
//	  result: Float32Array(fftLen),
//	  fftLen, directNs, fftNs, speedup, lagZeroIndex: number,
//	  kernel: string,
//	}
//
// or {error, panic} on failure.
func jsConvolve(opts js.Value) any {
	if !isObject(opts) {
		opts = js.Global().Get("Object").New()
	}

	req := parseConvolveRequest(opts)

	kernel64 := buildKernel(req.kernelName)
	if len(kernel64) > maxConvolveKernel {
		kernel64 = kernel64[:maxConvolveKernel]
	}

	a64 := make([]float64, req.n)
	generateSignalKind(a64, req.signal, signalParams{
		freqA: req.freqA,
		freqB: req.freqB,
		noise: req.noise,
		phase: req.phase,
		delta: req.delta,
		seed:  deterministicSeed(req.phase, req.n),
	})
	applyWindow(a64, req.window)

	a32 := make([]complex64, req.n)
	for i, v := range a64 {
		a32[i] = complex(float32(v), 0)
	}

	b32 := make([]complex64, len(kernel64))
	for i, v := range kernel64 {
		b32[i] = complex(float32(v), 0)
	}

	entry, err := planCache.getConvolver(req.n, len(kernel64), req.correlate)
	if err != nil {
		return errorResult(err)
	}

	var (
		fftLen int
		result []complex64
	)

	if req.correlate {
		fftLen = entry.corr.Len()
	} else {
		fftLen = entry.conv.Len()
	}

	result = make([]complex64, fftLen)

	start := time.Now()

	if req.correlate {
		err = entry.corr.CrossCorrelate(result, a32, b32)
	} else {
		err = entry.conv.Convolve(result, a32, b32)
	}

	fftNs := time.Since(start).Nanoseconds()

	if err != nil {
		return errorResult(err)
	}

	direct := make([]float64, fftLen)

	var directNs int64

	if req.n <= maxConvolveN && len(kernel64) <= maxConvolveKernel {
		startDirect := time.Now()

		if req.correlate {
			naiveCorrelate(direct, a64, kernel64)
		} else {
			naiveConvolve(direct, a64, kernel64)
		}

		directNs = time.Since(startDirect).Nanoseconds()
	}

	resultF32 := make([]float32, fftLen)
	for i, v := range result {
		resultF32[i] = real(v)
	}

	speedup := 0.0
	if fftNs > 0 && directNs > 0 {
		speedup = float64(directNs) / float64(fftNs)
	}

	lagZeroIndex := 0
	if req.correlate {
		lagZeroIndex = len(kernel64) - 1
	}

	return js.ValueOf(map[string]any{
		"result":       writeFloat32(outView(opts, "result"), resultF32),
		"fftLen":       fftLen,
		"directNs":     float64(directNs),
		"fftNs":        float64(fftNs),
		"speedup":      speedup,
		"lagZeroIndex": lagZeroIndex,
		"kernel":       req.kernelName,
	})
}

// buildKernel returns a small named FIR kernel, in float64. Unknown names
// fall back to lowpass.
func buildKernel(name string) []float64 {
	switch name {
	case "highpass":
		return highpassKernel(31, 0.15)
	case "edge":
		return []float64{-1, 2, -1}
	case "echo":
		return echoKernel(32, 12, 0.55)
	default: // "lowpass"
		return lowpassKernel(31, 0.15)
	}
}

// lowpassKernel returns a Hann-windowed-sinc low-pass FIR kernel with the
// given tap count and normalized cutoff (cycles/sample), normalized to unit
// DC gain.
func lowpassKernel(taps int, cutoff float64) []float64 {
	k := make([]float64, taps)
	mid := float64(taps-1) / 2
	sum := 0.0

	for i := 0; i < taps; i++ {
		x := float64(i) - mid

		var sinc float64
		if x == 0 {
			sinc = 2 * cutoff
		} else {
			sinc = math.Sin(2*math.Pi*cutoff*x) / (math.Pi * x)
		}

		w := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(taps-1)) // Hann
		k[i] = sinc * w
		sum += k[i]
	}

	if sum != 0 {
		for i := range k {
			k[i] /= sum
		}
	}

	return k
}

// highpassKernel derives a high-pass kernel from lowpassKernel by spectral
// inversion: negate the low-pass response and add back a unit impulse at its
// center tap.
func highpassKernel(taps int, cutoff float64) []float64 {
	lp := lowpassKernel(taps, cutoff)
	hp := make([]float64, taps)

	for i, v := range lp {
		hp[i] = -v
	}

	hp[(taps-1)/2] += 1

	return hp
}

// echoKernel returns an impulse at 0 plus a decayed repeat at delay,
// producing a single discrete echo when convolved with a signal.
func echoKernel(taps, delay int, decay float64) []float64 {
	k := make([]float64, taps)
	k[0] = 1

	if delay >= 0 && delay < taps {
		k[delay] = decay
	}

	return k
}

// naiveConvolve computes the O(len(a)*len(b)) direct linear convolution of a
// and b into dst (length len(a)+len(b)-1), as the reference the FFT-based
// path is checked and timed against.
func naiveConvolve(dst, a, b []float64) {
	for i := range dst {
		dst[i] = 0
	}

	for i, av := range a {
		if av == 0 {
			continue
		}

		for j, bv := range b {
			dst[i+j] += av * bv
		}
	}
}

// naiveCorrelate computes the direct cross-correlation of a and b into dst,
// matching Correlator.CrossCorrelate's convention (convolution of a with the
// reverse of b; inputs here are real, so no conjugate is needed).
func naiveCorrelate(dst, a, b []float64) {
	rev := make([]float64, len(b))
	for i, v := range b {
		rev[len(b)-1-i] = v
	}

	naiveConvolve(dst, a, rev)
}
