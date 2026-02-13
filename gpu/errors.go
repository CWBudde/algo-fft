package gpu

import "errors"

var (
	// ErrNoBackend is returned when no GPU backend is registered.
	ErrNoBackend = errors.New("algofft/gpu: no backend registered")

	// ErrBackendUnavailable is returned when the backend is registered but not available
	// on the current system (e.g., no device, driver missing).
	ErrBackendUnavailable = errors.New("algofft/gpu: backend unavailable")

	// ErrNotImplemented is returned by stubbed operations.
	ErrNotImplemented = errors.New("algofft/gpu: not implemented")

	// ErrInvalidLength is returned for invalid plan sizes.
	ErrInvalidLength = errors.New("algofft/gpu: invalid length")

	// ErrNilSlice is returned when dst or src is nil.
	ErrNilSlice = errors.New("algofft/gpu: nil slice")

	// ErrLengthMismatch is returned when dst or src lengths are not as required.
	ErrLengthMismatch = errors.New("algofft/gpu: length mismatch")
)
