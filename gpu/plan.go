package gpu

// Plan is a GPU-backed FFT plan for a specific size and precision.
//
// The plan owns GPU buffers and streams and is safe for concurrent use
// only if the underlying backend is thread-safe.
type Plan[T Complex] struct {
	n         int
	precision PrecisionKind
	ctx       Context
	streams   []Stream
	options   PlanOptions
	impl      PlanImpl
}

// NewPlan creates a GPU plan using the registered backend.
func NewPlan[T Complex](n int, opts PlanOptions) (*Plan[T], error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}

	backend := getBackend()
	if backend == nil {
		return nil, ErrNoBackend
	}

	if !backend.Available() {
		return nil, ErrBackendUnavailable
	}

	ctx, err := backend.NewContext(opts.DeviceIndex)
	if err != nil {
		return nil, err
	}

	precision := PrecisionComplex64
	var zero T
	switch any(zero).(type) {
	case complex128:
		precision = PrecisionComplex128
	case complex64:
		precision = PrecisionComplex64
	}

	streamCount := opts.StreamCount
	if streamCount <= 0 {
		streamCount = 1
	}

	streams := make([]Stream, 0, streamCount)
	for i := 0; i < streamCount; i++ {
		stream, err := ctx.NewStream()
		if err != nil {
			for _, s := range streams {
				_ = s.Close()
			}
			_ = ctx.Close()
			return nil, err
		}
		streams = append(streams, stream)
	}

	impl, err := ctx.NewFFTPlan(n, precision, opts)
	if err != nil {
		for _, s := range streams {
			_ = s.Close()
		}
		_ = ctx.Close()
		return nil, err
	}

	return &Plan[T]{
		n:         n,
		precision: precision,
		ctx:       ctx,
		streams:   streams,
		options:   opts,
		impl:      impl,
	}, nil
}

// Len returns the FFT length (number of complex samples) for this Plan.
func (p *Plan[T]) Len() int {
	if p == nil {
		return 0
	}
	return p.n
}

// Precision returns the plan precision.
func (p *Plan[T]) Precision() PrecisionKind {
	if p == nil {
		return PrecisionComplex64
	}
	return p.precision
}

// Forward computes the forward FFT on the GPU.
// This is a stub until a backend-specific execution path is implemented.
func (p *Plan[T]) Forward(dst, src []T) error {
	if p == nil {
		return ErrNotImplemented
	}
	if dst == nil || src == nil {
		return ErrNilSlice
	}
	if len(dst) < p.n || len(src) < p.n {
		return ErrLengthMismatch
	}
	if p.impl == nil {
		return ErrNotImplemented
	}
	return p.impl.Forward(dst, src)
}

// Inverse computes the inverse FFT on the GPU.
// This is a stub until a backend-specific execution path is implemented.
func (p *Plan[T]) Inverse(dst, src []T) error {
	if p == nil {
		return ErrNotImplemented
	}
	if dst == nil || src == nil {
		return ErrNilSlice
	}
	if len(dst) < p.n || len(src) < p.n {
		return ErrLengthMismatch
	}
	if p.impl == nil {
		return ErrNotImplemented
	}
	return p.impl.Inverse(dst, src)
}

// ForwardInPlace computes the forward FFT in-place.
func (p *Plan[T]) ForwardInPlace(data []T) error {
	return p.Forward(data, data)
}

// InverseInPlace computes the inverse FFT in-place.
func (p *Plan[T]) InverseInPlace(data []T) error {
	return p.Inverse(data, data)
}

// Close releases GPU resources associated with the plan.
func (p *Plan[T]) Close() error {
	if p == nil {
		return nil
	}
	if p.impl != nil {
		_ = p.impl.Close()
		p.impl = nil
	}
	var firstErr error
	for _, s := range p.streams {
		if s == nil {
			continue
		}
		if err := s.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	p.streams = nil
	if p.ctx != nil {
		if err := p.ctx.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		p.ctx = nil
	}
	return firstErr
}
