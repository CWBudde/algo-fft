package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// convolveRealT computes the linear convolution of a and b using real FFTs for
// the float type F and its matching complex type C. The dst slice must have
// length len(a)+len(b)-1.
func convolveRealT[F Float, C Complex](dst, a, b []F) error {
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

	fftLen := max(m.NextPowerOfTwo(convLen), 2)

	plan, err := NewPlanRealT[F, C](fftLen)
	if err != nil {
		return err
	}

	aPadded := make([]F, fftLen)
	bPadded := make([]F, fftLen)

	copy(aPadded, a)
	copy(bPadded, b)

	aFreq := make([]C, plan.SpectrumLen())
	bFreq := make([]C, plan.SpectrumLen())

	err = plan.Forward(aFreq, aPadded)
	if err != nil {
		return err
	}

	err = plan.Forward(bFreq, bPadded)
	if err != nil {
		return err
	}

	fft.ComplexMulArrayInPlace(aFreq, bFreq)

	time := make([]F, fftLen)

	err = plan.Inverse(time, aFreq)
	if err != nil {
		return err
	}

	copy(dst, time[:convLen])

	return nil
}

// ConvolveReal computes the linear convolution of a and b using real FFTs.
// The dst slice must have length len(a)+len(b)-1.
func ConvolveReal(dst, a, b []float32) error {
	return convolveRealT[float32, complex64](dst, a, b)
}

// ConvolveReal64 computes the linear convolution of a and b using double-precision
// real FFTs. The dst slice must have length len(a)+len(b)-1.
func ConvolveReal64(dst, a, b []float64) error {
	return convolveRealT[float64, complex128](dst, a, b)
}
