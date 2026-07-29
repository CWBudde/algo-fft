//go:build js && wasm

package main

import "math"

// windowKind selects an analysis window. The library itself has no window
// functions (verified: zero hits repo-wide outside tests and unrelated
// prose), so this is entirely demo-local.
//
// All windows here use the PERIODIC (DFT-even) convention w[i] = f(2*pi*i/N),
// not the symmetric N-1 denominator. The periodic form is the correct one for
// spectral analysis: the symmetric form is meant for FIR filter design, where
// the window must be exactly mirror-symmetric about its center sample, and
// using it here would leave a small asymmetry in the analyzed spectrum.
type windowKind uint8

const (
	windowRect windowKind = iota
	windowHann
	windowHamming
	windowBlackman
)

// windowKindFromString maps the JS-visible window name onto the enum,
// defaulting to the rectangular (no-op) window.
func windowKindFromString(s string) windowKind {
	switch s {
	case "hann":
		return windowHann
	case "hamming":
		return windowHamming
	case "blackman":
		return windowBlackman
	default:
		return windowRect
	}
}

// windowName returns the JS-visible name of k, the inverse of
// windowKindFromString.
func windowName(k windowKind) string {
	switch k {
	case windowHann:
		return "hann"
	case windowHamming:
		return "hamming"
	case windowBlackman:
		return "blackman"
	default:
		return "rect"
	}
}

// windowNames lists every window kind in declaration order, for the UI's
// selector.
func windowNames() []string {
	return []string{
		windowName(windowRect),
		windowName(windowHann),
		windowName(windowHamming),
		windowName(windowBlackman),
	}
}

// windowValue returns the window's periodic-convention value at sample i of
// an n-sample frame.
func windowValue(i, n int, k windowKind) float64 {
	if n <= 0 {
		return 1
	}

	switch k {
	case windowHann:
		return 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	case windowHamming:
		return 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n))
	case windowBlackman:
		x := 2 * math.Pi * float64(i) / float64(n)

		return 0.42 - 0.5*math.Cos(x) + 0.08*math.Cos(2*x)
	default: // windowRect
		return 1
	}
}

// windowShape fills dst with the window's shape (not applied to any signal),
// for the UI overlay.
func windowShape(dst []float64, k windowKind) {
	n := len(dst)

	for i := range dst {
		dst[i] = windowValue(i, n, k)
	}
}

// applyWindow multiplies dst in place by the window k's shape and returns the
// coherent gain (mean window value). Callers should divide windowed
// amplitudes by the coherent gain to keep them comparable across windows —
// rect has a gain of 1, Hann/Hamming/Blackman all attenuate the mean level.
func applyWindow(dst []float64, k windowKind) (coherentGain float64) {
	n := len(dst)
	if n == 0 {
		return 1
	}

	sum := 0.0

	for i := range dst {
		w := windowValue(i, n, k)
		dst[i] *= w
		sum += w
	}

	return sum / float64(n)
}
