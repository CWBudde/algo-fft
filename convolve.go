package algofft

import "github.com/cwbudde/algo-fft/internal/fft"

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

	plan, err := NewPlanT[T](convLen)
	if err != nil {
		return err
	}

	aPadded := make([]T, convLen)
	bPadded := make([]T, convLen)

	copy(aPadded, a)
	copy(bPadded, b)

	aFreq := make([]T, convLen)
	bFreq := make([]T, convLen)

	err = plan.Forward(aFreq, aPadded)
	if err != nil {
		return err
	}

	err = plan.Forward(bFreq, bPadded)
	if err != nil {
		return err
	}

	fft.ComplexMulArrayInPlace(aFreq, bFreq)

	time := make([]T, convLen)

	err = plan.Inverse(time, aFreq)
	if err != nil {
		return err
	}

	copy(dst, time)

	return nil
}

// Convolve computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
func Convolve(dst, a, b []complex64) error {
	return convolveT(dst, a, b)
}

// Convolve128 computes the linear convolution of a and b using FFTs.
// The dst slice must have length len(a)+len(b)-1.
func Convolve128(dst, a, b []complex128) error {
	return convolveT(dst, a, b)
}
