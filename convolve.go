package algofft

import (
	stdmath "math"

	"github.com/cwbudde/algo-fft/internal/fft"
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
// cheaper, so the pad candidates are costed exactly like bluesteinPadSize
// (next power of two vs next 5-smooth size, m·log2(m) with the measured
// mixed-radix penalty).
func fastConvolutionLength(convLen int) int {
	if convLen > maxBluesteinLength {
		// The pad arithmetic below would overflow; keep the exact length and
		// let plan creation report the same error it does today.
		return convLen
	}

	if m.IsPowerOf2(convLen) || planner.MixedRadixEligible(convLen) {
		return convLen
	}

	pow2 := m.NextPowerOfTwo(convLen)

	smooth := m.NextHighlyComposite(convLen)
	if smooth >= pow2 || smooth < 2 {
		return pow2
	}

	costPow2 := float64(pow2) * stdmath.Log2(float64(pow2))
	costSmooth := bluesteinSubFFTPenalty * float64(smooth) * stdmath.Log2(float64(smooth))

	if costSmooth < costPow2 {
		return smooth
	}

	return pow2
}

// convolveT computes the linear convolution of a and b for the complex type T.
// The dst slice must have length len(a)+len(b)-1.
func convolveT[T Complex](dst, a, b []T) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) == 0 || len(b) == 0 {
		return ErrInvalidLength
	}

	convLen := len(a) + len(b) - 1
	if len(dst) != convLen {
		return ErrLengthMismatch
	}

	fftLen := fastConvolutionLength(convLen)

	plan, err := NewPlanT[T](fftLen)
	if err != nil {
		return err
	}

	aPadded := make([]T, fftLen)
	bPadded := make([]T, fftLen)

	copy(aPadded, a)
	copy(bPadded, b)

	aFreq := make([]T, fftLen)
	bFreq := make([]T, fftLen)

	err = plan.Forward(aFreq, aPadded)
	if err != nil {
		return err
	}

	err = plan.Forward(bFreq, bPadded)
	if err != nil {
		return err
	}

	fft.ComplexMulArrayInPlace(aFreq, bFreq)

	time := make([]T, fftLen)

	err = plan.Inverse(time, aFreq)
	if err != nil {
		return err
	}

	copy(dst, time[:convLen])

	return nil
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
