package kernels

// SameSlice checks if two slices point to the same underlying array and have the same length.
// Useful for detecting in-place operations.
func SameSlice[T any](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}

	if len(a) == 0 {
		return true
	}

	return &a[0] == &b[0]
}

// sameSlice is a private alias for internal use.
func sameSlice[T any](a, b []T) bool {
	return SameSlice(a, b)
}
