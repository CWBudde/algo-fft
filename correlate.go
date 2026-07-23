package algofft

// crossCorrelateT computes the full cross-correlation of a and b for the complex
// type T. The dst slice must have length len(a)+len(b)-1.
//
// This is the one-shot entry point to the correlation pipeline owned by
// Correlator: it validates, builds a throwaway Correlator, and runs it once.
func crossCorrelateT[T Complex](dst, a, b []T) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) == 0 || len(b) == 0 {
		return ErrInvalidLength
	}

	if len(dst) != len(a)+len(b)-1 {
		return ErrLengthMismatch
	}

	corr, err := NewCorrelator[T](len(a), len(b))
	if err != nil {
		return err
	}

	return corr.CrossCorrelate(dst, a, b)
}

// Correlate computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func Correlate(dst, a, b []complex64) error {
	return crossCorrelateT(dst, a, b)
}

// CrossCorrelate computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
//
// Each call creates a fresh FFT plan; for repeated correlations of
// same-length inputs, use a Correlator to reuse the plan and buffers.
func CrossCorrelate(dst, a, b []complex64) error {
	return crossCorrelateT(dst, a, b)
}

// AutoCorrelate computes the full auto-correlation of a.
// The dst slice must have length 2*len(a)-1.
// Output index k corresponds to lag k-(len(a)-1).
func AutoCorrelate(dst, a []complex64) error {
	return crossCorrelateT(dst, a, a)
}

// Correlate128 computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func Correlate128(dst, a, b []complex128) error {
	return crossCorrelateT(dst, a, b)
}

// CrossCorrelate128 computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func CrossCorrelate128(dst, a, b []complex128) error {
	return crossCorrelateT(dst, a, b)
}

// AutoCorrelate128 computes the full auto-correlation of a.
// The dst slice must have length 2*len(a)-1.
// Output index k corresponds to lag k-(len(a)-1).
func AutoCorrelate128(dst, a []complex128) error {
	return crossCorrelateT(dst, a, a)
}
