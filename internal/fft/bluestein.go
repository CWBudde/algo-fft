package fft

import (
	"strconv"

	"github.com/cwbudde/algo-fft/internal/kernels"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// ComputeBluesteinFilter computes the frequency-domain Bluestein filter for
// padded size m. Power-of-two m runs the radix-2 DIT sub-FFT; any other m the
// pad chooser selects runs through the mixed-radix engine.
func ComputeBluesteinFilter[T Complex](n, m int, chirp []T, twiddles []T, scratch []T) []T {
	if mathpkg.IsPowerOf2(m) {
		return kernels.ComputeBluesteinFilter[T](n, m, chirp, twiddles, scratch)
	}

	b := kernels.BuildBluesteinSequence(n, m, chirp)
	mustMixedRadix(mixedRadixForward(b, b, twiddles, scratch), m)

	return b
}

// BluesteinConvolution performs the cyclic convolution y = x * b via a padded
// sub-FFT of size m = len(filter). Power-of-two m uses the radix-2 DIT kernels
// with the precomputed bitrev table; any other m dispatches to the mixed-radix
// engine (which ignores bitrev).
func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T, bitrev []int) {
	m := len(filter)
	if mathpkg.IsPowerOf2(m) {
		kernels.BluesteinConvolution[T](dst, x, filter, twiddles, scratch, bitrev)
		return
	}

	mustMixedRadix(mixedRadixForward(dst, x, twiddles, scratch), m)

	ComplexMulArrayInPlace(dst[:m], filter)

	mustMixedRadix(mixedRadixInverse(dst, dst, twiddles, scratch), m)
}

// mustMixedRadix panics when the mixed-radix engine rejects a Bluestein
// sub-FFT size. Plan construction only selects padded sizes the engine
// schedules unconditionally (see padShapes), so a failure here is a
// planner/engine contract violation — returning would leave dst partially
// written and surface as a silent wrong answer.
func mustMixedRadix(ok bool, m int) {
	if !ok {
		panic("algofft: mixed-radix engine rejected Bluestein sub-FFT size " + strconv.Itoa(m) +
			" (planner/engine contract violation)")
	}
}
