package algoforge

// ForwardStrided computes the forward FFT on strided input/output data.
//
// The stride parameter specifies the distance between consecutive elements.
// For example, stride=numCols transforms a matrix column in row-major storage.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidStride if stride < 1 or overflows index computation.
// Returns ErrLengthMismatch if slices are too short for the given stride.
func (p *Plan[T]) ForwardStrided(dst, src []T, stride int) error {
	return p.transformStrided(dst, src, stride, false)
}

// InverseStrided computes the inverse FFT on strided input/output data.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidStride if stride < 1 or overflows index computation.
// Returns ErrLengthMismatch if slices are too short for the given stride.
func (p *Plan[T]) InverseStrided(dst, src []T, stride int) error {
	return p.transformStrided(dst, src, stride, true)
}

// TransformStrided computes either forward or inverse FFT based on the inverse flag.
// This is a convenience wrapper over ForwardStrided/InverseStrided.
func (p *Plan[T]) TransformStrided(dst, src []T, stride int, inverse bool) error {
	return p.transformStrided(dst, src, stride, inverse)
}

func (p *Plan[T]) transformStrided(dst, src []T, stride int, inverse bool) error {
	err := p.validateStridedSlices(dst, src, stride)
	if err != nil {
		return err
	}

	if stride == 1 {
		if inverse {
			return p.Inverse(dst[:p.n], src[:p.n])
		}

		return p.Forward(dst[:p.n], src[:p.n])
	}

	buffer := p.stridedScratch[:p.n]
	for i := 0; i < p.n; i++ {
		buffer[i] = src[i*stride]
	}

	if inverse {
		if err := p.Inverse(buffer, buffer); err != nil {
			return err
		}
	} else {
		if err := p.Forward(buffer, buffer); err != nil {
			return err
		}
	}

	for i := 0; i < p.n; i++ {
		dst[i*stride] = buffer[i]
	}

	return nil
}

func (p *Plan[T]) validateStridedSlices(dst, src []T, stride int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if stride < 1 {
		return ErrInvalidStride
	}

	if p.n == 0 {
		return ErrLengthMismatch
	}

	if stride == 1 {
		if len(dst) < p.n || len(src) < p.n {
			return ErrLengthMismatch
		}

		return nil
	}

	maxInt := int(^uint(0) >> 1)
	maxIndex := p.n - 1
	if maxIndex > (maxInt-1)/stride {
		return ErrInvalidStride
	}

	required := 1 + maxIndex*stride
	if len(dst) < required || len(src) < required {
		return ErrLengthMismatch
	}

	return nil
}
