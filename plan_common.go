package algofft

// This file holds the small helpers shared by all plan types: precision
// naming for String() methods, dst/src argument validation, and the in-place
// transform dispatch used by the multi-dimensional engines.

// complexTypeName returns the precision name ("complex64" or "complex128")
// for the complex type T, for use in String() methods.
func complexTypeName[T Complex]() string {
	var zero T
	if _, ok := any(zero).(complex128); ok {
		return precisionNameComplex128
	}

	return precisionNameComplex64
}

// validateDstSrc checks the dst/src pair every transform method receives:
// both slices must be non-nil and have exactly the expected lengths. The two
// element types are independent so real plans (float in, complex out) can use
// the same helper as complex plans.
func validateDstSrc[D, S any](dst []D, src []S, wantDst, wantSrc int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(dst) != wantDst || len(src) != wantSrc {
		return ErrLengthMismatch
	}

	return nil
}

// transformSliceInPlace runs the plan's forward or inverse transform in place
// on data, selecting the direction at the call site. Used by the
// multi-dimensional engines, which apply the same axis loop in both
// directions.
func transformSliceInPlace[T Complex](plan *Plan[T], data []T, forward bool) error {
	if forward {
		return plan.ForwardInPlace(data)
	}

	return plan.InverseInPlace(data)
}
