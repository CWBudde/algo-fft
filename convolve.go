package algofft

import (
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// fastConvolutionLength returns the FFT length used to compute a linear
// convolution or correlation of logical length convLen. Any cyclic length
// >= convLen works (the inputs are zero-padded, so no wraparound reaches the
// first convLen samples), which frees the FFT size from the awkward lengths
// convolution naturally produces.
//
// Lengths the engine executes exactly (powers of two, and the mixed-radix
// smooth lengths that pass their win gate) are kept unchanged — padding those
// is not a measured win on the purego build. Anything else would route to the
// arbitrary-length fallbacks (Rader/Bluestein), which run two-plus sub-FFTs
// of at least comparable size per transform; a single padded FFT is strictly
// cheaper, so the pad candidates are costed via the shared pad model
// (cheapestPaddedLength, also used for Bluestein sub-FFT sizing).
func fastConvolutionLength(convLen int) int {
	if convLen > maxBluesteinLength {
		// The pad arithmetic below would overflow; keep the exact length and
		// let plan creation report the same error it does today.
		return convLen
	}

	if m.IsPowerOf2(convLen) || planner.MixedRadixEligible(convLen) {
		return convLen
	}

	return cheapestPaddedLength(convLen)
}

// convolveT computes the linear convolution of a and b for the complex type T.
// The dst slice must have length len(a)+len(b)-1.
//
// This is the one-shot entry point to the convolution pipeline owned by
// Convolver: it validates, builds a throwaway Convolver, and runs it once.
func convolveT[T Complex](dst, a, b []T) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) == 0 || len(b) == 0 {
		return ErrInvalidLength
	}

	if len(dst) != len(a)+len(b)-1 {
		return ErrLengthMismatch
	}

	conv, err := NewConvolver[T](len(a), len(b))
	if err != nil {
		return err
	}

	return conv.Convolve(dst, a, b)
}

// Convolve computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
//
// Each call creates a fresh FFT plan; for repeated convolutions of
// same-length inputs, use a Convolver to reuse the plan and buffers.
func Convolve(dst, a, b []complex64) error {
	return convolveT(dst, a, b)
}

// Convolve128 computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
func Convolve128(dst, a, b []complex128) error {
	return convolveT(dst, a, b)
}
