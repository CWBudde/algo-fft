package algofft

// This file holds the transform entry points of the 1D Plan: validation and
// scratch acquisition live here, while the strategy-specific work happens in
// the plan's executor (see plan_exec*.go).

func (p *Plan[T]) getScratch() ([]T, []T, []T, *scratchSet[T]) {
	if p.scratch != nil {
		return p.scratch, p.stridedScratch, p.subScratch, nil
	}

	s := p.scratchPool.get()

	return s.scratch, s.stridedScratch, s.subScratch, s
}

// validateSlices checks that dst and src are valid for this Plan.
func (p *Plan[T]) validateSlices(dst, src []T) error {
	return validateDstSrc(dst, src, p.n, p.n)
}

// Forward computes the forward (time-to-frequency) FFT.
//
// The transform is computed as:
//
//	X[k] = Σ x[n] * exp(-2πink/N) for k = 0..N-1
//
// dst and src must have length equal to Plan.Len().
// dst and src may point to the same slice for in-place operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match Plan dimensions.
func (p *Plan[T]) Forward(dst, src []T) error {
	err := p.validateSlices(dst, src)
	if err != nil {
		return err
	}

	scratch, _, sub, set := p.getScratch()
	if set != nil {
		defer p.scratchPool.put(set)
	}

	// Zero-dispatch codelet fast path (see the field comment in plan.go); a
	// bailing codelet falls through to the executor, which retries it before
	// its kernel fallback.
	if p.forwardCodelet != nil && p.forwardCodelet(dst, src, p.codeletTwiddleForward, scratch) {
		return nil
	}

	p.exec.forward(dst, src, scratch, sub)

	return nil
}

// Inverse computes the inverse (frequency-to-time) FFT.
//
// The transform is computed as:
//
//	x[n] = (1/N) * Σ X[k] * exp(2πink/N) for n = 0..N-1
//
// dst and src must have length equal to Plan.Len().
// dst and src may point to the same slice for in-place operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match Plan dimensions.
func (p *Plan[T]) Inverse(dst, src []T) error {
	err := p.validateSlices(dst, src)
	if err != nil {
		return err
	}

	scratch, _, sub, set := p.getScratch()
	if set != nil {
		defer p.scratchPool.put(set)
	}

	// Zero-dispatch codelet fast path (see the field comment in plan.go); a
	// bailing codelet falls through to the executor, which retries it before
	// its kernel fallback.
	if p.inverseCodelet != nil && p.inverseCodelet(dst, src, p.codeletTwiddleInverse, scratch) {
		return nil
	}

	p.exec.inverse(dst, src, scratch, sub)

	return nil
}

// ForwardInPlace computes the forward FFT in-place, modifying the input slice directly.
//
// This is equivalent to Forward(data, data) but may be slightly more efficient.
//
// Returns ErrNilSlice if data is nil.
// Returns ErrLengthMismatch if slice length doesn't match Plan dimensions.
func (p *Plan[T]) ForwardInPlace(data []T) error {
	return p.Forward(data, data)
}

// InverseInPlace computes the inverse FFT in-place, modifying the input slice directly.
//
// This is equivalent to Inverse(data, data) but may be slightly more efficient.
//
// Returns ErrNilSlice if data is nil.
// Returns ErrLengthMismatch if slice length doesn't match Plan dimensions.
func (p *Plan[T]) InverseInPlace(data []T) error {
	return p.Inverse(data, data)
}

// ForwardUnsafe performs the forward FFT without any validation.
// This is a zero-overhead path for latency-critical workloads.
//
// REQUIREMENTS (caller must guarantee):
//   - dst and src are non-nil slices
//   - len(dst) >= p.Len() and len(src) >= p.Len()
//   - Plan has pre-allocated scratch (pooled or cloned plans)
//
// Violating these requirements causes undefined behavior or panic.
// Use Forward() for the safe, validated path.
func (p *Plan[T]) ForwardUnsafe(dst, src []T) {
	if p.forwardCodelet != nil && p.forwardCodelet(dst, src, p.codeletTwiddleForward, p.scratch) {
		return
	}

	p.exec.forward(dst, src, p.scratch, p.subScratch)
}

// InverseUnsafe performs the inverse FFT without any validation.
// This is a zero-overhead path for latency-critical workloads.
//
// REQUIREMENTS (caller must guarantee):
//   - dst and src are non-nil slices
//   - len(dst) >= p.Len() and len(src) >= p.Len()
//   - Plan has pre-allocated scratch (pooled or cloned plans)
//
// Violating these requirements causes undefined behavior or panic.
// Use Inverse() for the safe, validated path.
func (p *Plan[T]) InverseUnsafe(dst, src []T) {
	if p.inverseCodelet != nil && p.inverseCodelet(dst, src, p.codeletTwiddleInverse, p.scratch) {
		return
	}

	p.exec.inverse(dst, src, p.scratch, p.subScratch)
}

// Transform computes either forward or inverse FFT based on the inverse flag.
// This is a convenience wrapper over Forward/Inverse.
func (p *Plan[T]) Transform(dst, src []T, inverse bool) error {
	if inverse {
		return p.Inverse(dst, src)
	}

	return p.Forward(dst, src)
}

// ForwardBatch computes count forward FFTs on sequential data.
//
// The data layout is interleaved/sequential:
//   - FFT 0: src[0:n] → dst[0:n]
//   - FFT 1: src[n:2n] → dst[n:2n]
//   - FFT i: src[i*n:(i+1)*n] → dst[i*n:(i+1)*n]
//
// dst and src must have length >= count * Plan.Len().
// dst and src may point to the same slice for in-place batch operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidLength if count < 1.
// Returns ErrLengthMismatch if slice lengths are insufficient.
func (p *Plan[T]) ForwardBatch(dst, src []T, count int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if count < 1 {
		return ErrInvalidLength
	}

	required := count * p.n
	if len(dst) < required || len(src) < required {
		return ErrLengthMismatch
	}

	for i := range count {
		start := i * p.n

		end := start + p.n

		err := p.Forward(dst[start:end], src[start:end])
		if err != nil {
			return err
		}
	}

	return nil
}

// InverseBatch computes count inverse FFTs on sequential data.
//
// The data layout is interleaved/sequential:
//   - FFT 0: src[0:n] → dst[0:n]
//   - FFT 1: src[n:2n] → dst[n:2n]
//   - FFT i: src[i*n:(i+1)*n] → dst[i*n:(i+1)*n]
//
// dst and src must have length >= count * Plan.Len().
// dst and src may point to the same slice for in-place batch operation.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidLength if count < 1.
// Returns ErrLengthMismatch if slice lengths are insufficient.
func (p *Plan[T]) InverseBatch(dst, src []T, count int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if count < 1 {
		return ErrInvalidLength
	}

	required := count * p.n
	if len(dst) < required || len(src) < required {
		return ErrLengthMismatch
	}

	for i := range count {
		start := i * p.n

		end := start + p.n

		err := p.Inverse(dst[start:end], src[start:end])
		if err != nil {
			return err
		}
	}

	return nil
}
