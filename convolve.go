package algoforge

// Convolve computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
func Convolve(dst, a, b []complex64) error {
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

	plan, err := NewPlanT[complex64](convLen)
	if err != nil {
		return err
	}

	aPadded := make([]complex64, convLen)
	bPadded := make([]complex64, convLen)

	copy(aPadded, a)
	copy(bPadded, b)

	aFreq := make([]complex64, convLen)
	bFreq := make([]complex64, convLen)

	if err := plan.Forward(aFreq, aPadded); err != nil {
		return err
	}

	if err := plan.Forward(bFreq, bPadded); err != nil {
		return err
	}

	for i := range aFreq {
		aFreq[i] *= bFreq[i]
	}

	time := make([]complex64, convLen)
	if err := plan.Inverse(time, aFreq); err != nil {
		return err
	}

	copy(dst, time)

	return nil
}
