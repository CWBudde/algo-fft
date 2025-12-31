package algofft

// Convolve computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
//
//nolint:cyclop
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

	time := make([]complex64, convLen)

	err = plan.Inverse(time, aFreq)
	if err != nil {
		return err
	}

	copy(dst, time)

	return nil
}

// Convolve128 computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
//
//nolint:cyclop
func Convolve128(dst, a, b []complex128) error {
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

	plan, err := NewPlanT[complex128](convLen)
	if err != nil {
		return err
	}

	aPadded := make([]complex128, convLen)
	bPadded := make([]complex128, convLen)

	copy(aPadded, a)
	copy(bPadded, b)

	aFreq := make([]complex128, convLen)
	bFreq := make([]complex128, convLen)

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

	time := make([]complex128, convLen)

	err = plan.Inverse(time, aFreq)
	if err != nil {
		return err
	}

	copy(dst, time)

	return nil
}
