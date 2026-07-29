//go:build js && wasm

package main

import (
	"math"
	"math/rand"
)

// signalKind selects a synthetic waveform generator. Frequencies throughout
// this file are in cycles-per-frame, so a tone at frequency f lands exactly
// on FFT bin f (for integer f) — the demo's whole point is that bin index and
// requested frequency read the same.
type signalKind uint8

const (
	sigTone signalKind = iota
	sigChirp
	sigImpulse
	sigSquare
	sigSaw
	sigAM
	sigFM
	sigCloseTones
)

// signalKindFromString maps the JS-visible signal name onto the enum,
// defaulting to the two-tone "tone" generator.
func signalKindFromString(s string) signalKind {
	switch s {
	case "chirp":
		return sigChirp
	case "impulse":
		return sigImpulse
	case "square":
		return sigSquare
	case "saw":
		return sigSaw
	case "am":
		return sigAM
	case "fm":
		return sigFM
	case "close":
		return sigCloseTones
	default:
		return sigTone
	}
}

// signalName returns the JS-visible name of k, the inverse of
// signalKindFromString.
func signalName(k signalKind) string {
	switch k {
	case sigChirp:
		return "chirp"
	case sigImpulse:
		return "impulse"
	case sigSquare:
		return "square"
	case sigSaw:
		return "saw"
	case sigAM:
		return "am"
	case sigFM:
		return "fm"
	case sigCloseTones:
		return "close"
	default:
		return "tone"
	}
}

// signalNames lists every signal kind in declaration order, for the UI's
// selector.
func signalNames() []string {
	return []string{
		signalName(sigTone),
		signalName(sigChirp),
		signalName(sigImpulse),
		signalName(sigSquare),
		signalName(sigSaw),
		signalName(sigAM),
		signalName(sigFM),
		signalName(sigCloseTones),
	}
}

// signalParams carries every knob a generator can read. Not every generator
// uses every field: FM repurposes noise as a modulation index (see below),
// impulse ignores everything but freqA, and only close reads delta.
type signalParams struct {
	freqA float64
	freqB float64
	noise float64
	phase float64
	delta float64 // sigCloseTones: fractional-bin separation from freqA
	seed  int64
}

// defaultCloseDelta is used when a caller asks for sigCloseTones without
// setting delta explicitly.
const defaultCloseDelta = 0.5

// generateSignalKind fills dst (length n, one frame) with the waveform named
// by kind, deterministically from p.seed.
func generateSignalKind(dst []float64, kind signalKind, p signalParams) {
	n := len(dst)
	if n == 0 {
		return
	}

	rng := rand.New(rand.NewSource(p.seed)) //nolint:gosec // deterministic demo noise

	switch kind {
	case sigChirp:
		generateChirp(dst, p, rng)
	case sigImpulse:
		generateImpulse(dst, p)
	case sigSquare:
		generateSquare(dst, p, rng)
	case sigSaw:
		generateSaw(dst, p, rng)
	case sigAM:
		generateAM(dst, p, rng)
	case sigFM:
		generateFM(dst, p)
	case sigCloseTones:
		generateCloseTones(dst, p, rng)
	default: // sigTone
		generateTone(dst, p, rng)
	}
}

// addNoise adds uniform noise in [-p.noise, +p.noise] to dst[i], deterministic
// via rng. A no-op when p.noise <= 0.
func addNoise(dst []float64, i int, noise float64, rng *rand.Rand) {
	if noise > 0 {
		dst[i] += (rng.Float64()*2 - 1) * noise
	}
}

// generateTone is the original two-sine-plus-noise waveform, kept
// behaviourally identical to what analyze.go used to generate inline so nothing
// regresses for callers that do not pass a signal kind.
func generateTone(dst []float64, p signalParams, rng *rand.Rand) {
	n := len(dst)
	for i := range dst {
		t := float64(i) / float64(n)
		s := math.Sin(2*math.Pi*p.freqA*t+p.phase) +
			0.65*math.Sin(2*math.Pi*p.freqB*t+p.phase*0.7)
		dst[i] = s

		addNoise(dst, i, p.noise, rng)
	}
}

// generateChirp sweeps linearly from freqA to freqB across the frame. The
// instantaneous phase is the integral of instantaneous frequency:
// freqA*t + 0.5*(freqB-freqA)*t^2 cycles.
func generateChirp(dst []float64, p signalParams, rng *rand.Rand) {
	n := len(dst)
	for i := range dst {
		t := float64(i) / float64(n)
		cycles := p.freqA*t + 0.5*(p.freqB-p.freqA)*t*t
		dst[i] = math.Sin(2*math.Pi*cycles + p.phase)

		addNoise(dst, i, p.noise, rng)
	}
}

// generateImpulse places a unit spike at index int(freqA) mod n and zeroes
// everything else. Its DFT is exactly flat in magnitude (|X[k]| = 1 for all
// k) and its phase is a linear ramp with slope equal to the spike's index —
// deliberately left free of noise so it stays the cleanest correctness check
// in the demo.
func generateImpulse(dst []float64, p signalParams) {
	n := len(dst)

	idx := int(p.freqA) % n
	if idx < 0 {
		idx += n
	}

	for i := range dst {
		dst[i] = 0
	}

	dst[idx] = 1
}

// generateSquare is an ideal square wave at freqA: sign(sin(...)). Its
// Fourier series has energy at odd harmonics of freqA only.
func generateSquare(dst []float64, p signalParams, rng *rand.Rand) {
	n := len(dst)
	for i := range dst {
		t := float64(i) / float64(n)
		if math.Sin(2*math.Pi*p.freqA*t+p.phase) >= 0 {
			dst[i] = 1
		} else {
			dst[i] = -1
		}

		addNoise(dst, i, p.noise, rng)
	}
}

// generateSaw is an ideal sawtooth at freqA, harmonic-rich (all harmonics,
// not just odd ones).
func generateSaw(dst []float64, p signalParams, rng *rand.Rand) {
	n := len(dst)
	for i := range dst {
		t := float64(i) / float64(n)
		phase := p.freqA*t + p.phase/(2*math.Pi)
		frac := phase - math.Floor(phase)
		dst[i] = 2*frac - 1

		addNoise(dst, i, p.noise, rng)
	}
}

// generateAM amplitude-modulates a freqA carrier with a freqB modulator.
// Expanding the product with the standard identity
//
//	sin(fB t) * sin(fA t) = 0.5*[cos((fA-fB)t) - cos((fA+fB)t)]
//
// shows the result is exactly a carrier at freqA plus two symmetric
// sidebands at freqA-freqB and freqA+freqB — textbook AM.
func generateAM(dst []float64, p signalParams, rng *rand.Rand) {
	n := len(dst)
	for i := range dst {
		t := float64(i) / float64(n)
		mod := 1 + math.Sin(2*math.Pi*p.freqB*t)
		dst[i] = 0.5 * mod * math.Sin(2*math.Pi*p.freqA*t+p.phase)

		addNoise(dst, i, p.noise, rng)
	}
}

// fmMaxIndex is the upper end of the modulation-index sweep the noise slider
// is remapped onto for FM, chosen so the slider's full range sweeps through
// the first carrier null at the Bessel zero beta ~= 2.405.
const fmMaxIndex = 10.0

// generateFM frequency-modulates a freqA carrier with a freqB modulator at
// modulation index beta = noise * fmMaxIndex, so sweeping the (repurposed)
// noise slider from 0 to 1 sweeps beta from 0 to 10 and crosses the carrier
// null at beta ~= 2.405.
func generateFM(dst []float64, p signalParams) {
	n := len(dst)
	beta := p.noise * fmMaxIndex

	for i := range dst {
		t := float64(i) / float64(n)
		dst[i] = math.Sin(2*math.Pi*p.freqA*t + beta*math.Sin(2*math.Pi*p.freqB*t) + p.phase)
	}
}

// generateCloseTones sums two tones at freqA and freqA+delta, where delta is
// a FRACTIONAL bin separation (default 0.5). Combined with the window
// selector this demonstrates the Rayleigh resolution limit: rect resolves
// closer-spaced tones but leaks, Blackman leaks less but merges sooner.
func generateCloseTones(dst []float64, p signalParams, rng *rand.Rand) {
	n := len(dst)

	delta := p.delta
	if delta == 0 {
		delta = defaultCloseDelta
	}

	for i := range dst {
		t := float64(i) / float64(n)
		s := math.Sin(2*math.Pi*p.freqA*t+p.phase) +
			math.Sin(2*math.Pi*(p.freqA+delta)*t+p.phase)
		dst[i] = s

		addNoise(dst, i, p.noise, rng)
	}
}

// deterministicSeed derives the demo's noise seed from phase and frame
// length, matching the formula the original inline tone generator used, so
// results are reproducible across reloads for the same request.
func deterministicSeed(phase float64, n int) int64 {
	return int64(math.Round(phase*1000)) + int64(n)*37
}
