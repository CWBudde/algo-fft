package algofft

import m "github.com/cwbudde/algo-fft/internal/math"

// ConvolveReal computes the linear convolution of a and b using real FFTs.
// The dst slice must have length len(a)+len(b)-1.
//
//nolint:cyclop
func ConvolveReal(dst, a, b []float32) error {
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

	fftLen := m.NextPowerOfTwo(convLen)
	if fftLen < 2 {
		fftLen = 2
	}

	plan, err := NewPlanReal(fftLen)
	if err != nil {
		return err
	}

	aPadded := make([]float32, fftLen)
	bPadded := make([]float32, fftLen)

	copy(aPadded, a)
	copy(bPadded, b)

	aFreq := make([]complex64, plan.SpectrumLen())
	bFreq := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(aFreq, aPadded)
	if err != nil {
		return err
	}

	err = plan.Forward(bFreq, bPadded)
	if err != nil {
		return err
	}

	for i := range aFreq {
		aFreq[i] *= bFreq[i]
	}

	time := make([]float32, fftLen)

	err = plan.Inverse(time, aFreq)
	if err != nil {
		return err
	}

	copy(dst, time[:convLen])

	return nil
}
