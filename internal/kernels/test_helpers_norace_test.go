//go:build !race

package kernels

// raceDetectorEnabled reports whether the race detector is active; see the
// race-tagged twin for why tests check it.
const raceDetectorEnabled = false
