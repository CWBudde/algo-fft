package algofft

// Correlate computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func Correlate(dst, a, b []complex64) error {
	return CrossCorrelate(dst, a, b)
}

// CrossCorrelate computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func CrossCorrelate(dst, a, b []complex64) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) == 0 || len(b) == 0 {
		return ErrInvalidLength
	}

	if len(dst) != len(a)+len(b)-1 {
		return ErrLengthMismatch
	}

	bRevConj := make([]complex64, len(b))
	for i := range b {
		v := b[len(b)-1-i]
		bRevConj[i] = complex(real(v), -imag(v))
	}

	return Convolve(dst, a, bRevConj)
}

// AutoCorrelate computes the full auto-correlation of a.
// The dst slice must have length 2*len(a)-1.
// Output index k corresponds to lag k-(len(a)-1).
func AutoCorrelate(dst, a []complex64) error {
	return CrossCorrelate(dst, a, a)
}

// Correlate128 computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func Correlate128(dst, a, b []complex128) error {
	return CrossCorrelate128(dst, a, b)
}

// CrossCorrelate128 computes the full cross-correlation of a and b.
// The dst slice must have length len(a)+len(b)-1.
// Output index k corresponds to lag k-(len(b)-1).
func CrossCorrelate128(dst, a, b []complex128) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) == 0 || len(b) == 0 {
		return ErrInvalidLength
	}

	if len(dst) != len(a)+len(b)-1 {
		return ErrLengthMismatch
	}

	bRevConj := make([]complex128, len(b))
	for i := range b {
		v := b[len(b)-1-i]
		bRevConj[i] = complex(real(v), -imag(v))
	}

	return Convolve128(dst, a, bRevConj)
}

// AutoCorrelate128 computes the full auto-correlation of a.
// The dst slice must have length 2*len(a)-1.
// Output index k corresponds to lag k-(len(a)-1).
func AutoCorrelate128(dst, a []complex128) error {
	return CrossCorrelate128(dst, a, a)
}
