//go:build race

package kernels

// raceDetectorEnabled reports whether the race detector is active, so tests
// can skip work that is only prohibitively slow under its instrumentation
// (e.g. the O(n²) naive reference DFT at n = 32768).
const raceDetectorEnabled = true
