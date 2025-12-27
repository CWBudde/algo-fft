package algofft

import "github.com/MeKo-Christian/algofft/internal/fft"

// ConvolveReal computes the linear convolution of a and b using real FFTs.
// The dst slice must have length len(a)+len(b)-1.
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

	fftLen := fft.NextPowerOfTwo(convLen)
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

	if err := plan.Forward(aFreq, aPadded); err != nil {
		return err
	}

	if err := plan.Forward(bFreq, bPadded); err != nil {
		return err
	}

	for i := range aFreq {
		aFreq[i] *= bFreq[i]
	}

	time := make([]float32, fftLen)
	if err := plan.Inverse(time, aFreq); err != nil {
		return err
	}

	copy(dst, time[:convLen])

	return nil
}
