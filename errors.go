package algoforge

import "errors"

// Sentinel errors returned by FFT operations.
var (
	// ErrInvalidLength is returned when the FFT size is not valid.
	// For basic plans, the length must be a positive power of 2.
	// Mixed-radix and Bluestein algorithms extend supported sizes.
	ErrInvalidLength = errors.New("algoforge: invalid FFT length")

	// ErrNilSlice is returned when a nil slice is passed to a transform method.
	ErrNilSlice = errors.New("algoforge: nil slice")

	// ErrLengthMismatch is returned when input/output slice sizes don't match
	// the Plan's expected dimensions.
	ErrLengthMismatch = errors.New("algoforge: slice length mismatch")

	// ErrInvalidStride is returned when a stride parameter is invalid
	// for the given data layout (e.g., stride < 1 or doesn't align with data).
	ErrInvalidStride = errors.New("algoforge: invalid stride")

	// ErrNotImplemented is returned for features that are not yet implemented.
	// This is a temporary error used during development.
	ErrNotImplemented = errors.New("algoforge: not implemented")
)
