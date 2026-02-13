package gpu

import (
	"fmt"

	algofft "github.com/cwbudde/algo-fft"
)

// MockBackend is a CPU-backed GPU backend for development and tests.
// It satisfies the GPU backend interfaces but executes on the CPU.
type MockBackend struct {
	device DeviceInfo
}

// NewMockBackend returns a mock backend with a single fake device.
func NewMockBackend() *MockBackend {
	return &MockBackend{
		device: DeviceInfo{
			Name:       "MockGPU",
			Vendor:     "algofft",
			Driver:     "mock",
			MemoryMB:   0,
			ComputeCap: "cpu",
		},
	}
}

func (b *MockBackend) Info() BackendInfo {
	return BackendInfo{
		Name:        "mock",
		Version:     "0.1",
		Description: "CPU-backed mock GPU backend",
	}
}

func (b *MockBackend) Available() bool {
	return true
}

func (b *MockBackend) Devices() ([]DeviceInfo, error) {
	return []DeviceInfo{b.device}, nil
}

func (b *MockBackend) NewContext(deviceIndex int) (Context, error) {
	if deviceIndex != 0 {
		return nil, fmt.Errorf("mock backend: device index %d out of range", deviceIndex)
	}
	return &mockContext{device: b.device}, nil
}

// RegisterMockBackend registers the mock backend as the active backend.
func RegisterMockBackend() {
	RegisterBackend(NewMockBackend())
}

type mockContext struct {
	device DeviceInfo
}

func (c *mockContext) Device() DeviceInfo {
	return c.device
}

func (c *mockContext) NewBuffer(elemCount int, precision PrecisionKind) (Buffer, error) {
	if elemCount < 0 {
		return nil, ErrInvalidLength
	}
	switch precision {
	case PrecisionComplex64:
		return &mockBuffer{
			precision: precision,
			len:       elemCount,
			data64:    make([]complex64, elemCount),
		}, nil
	case PrecisionComplex128:
		return &mockBuffer{
			precision: precision,
			len:       elemCount,
			data128:   make([]complex128, elemCount),
		}, nil
	default:
		return nil, ErrNotImplemented
	}
}

func (c *mockContext) NewStream() (Stream, error) {
	return &mockStream{}, nil
}

func (c *mockContext) NewFFTPlan(n int, precision PrecisionKind, _ PlanOptions) (PlanImpl, error) {
	if n < 1 {
		return nil, ErrInvalidLength
	}
	switch precision {
	case PrecisionComplex64:
		p, err := algofft.NewPlanT[complex64](n)
		if err != nil {
			return nil, err
		}
		return &mockPlan64{plan: p}, nil
	case PrecisionComplex128:
		p, err := algofft.NewPlanT[complex128](n)
		if err != nil {
			return nil, err
		}
		return &mockPlan128{plan: p}, nil
	default:
		return nil, ErrNotImplemented
	}
}

func (c *mockContext) Close() error {
	return nil
}

type mockBuffer struct {
	precision PrecisionKind
	len       int
	data64    []complex64
	data128   []complex128
}

func (b *mockBuffer) Len() int {
	return b.len
}

func (b *mockBuffer) Precision() PrecisionKind {
	return b.precision
}

func (b *mockBuffer) Upload(src any) error {
	switch b.precision {
	case PrecisionComplex64:
		data, ok := src.([]complex64)
		if !ok {
			return ErrNotImplemented
		}
		if len(data) < b.len {
			return ErrLengthMismatch
		}
		copy(b.data64, data[:b.len])
		return nil
	case PrecisionComplex128:
		data, ok := src.([]complex128)
		if !ok {
			return ErrNotImplemented
		}
		if len(data) < b.len {
			return ErrLengthMismatch
		}
		copy(b.data128, data[:b.len])
		return nil
	default:
		return ErrNotImplemented
	}
}

func (b *mockBuffer) Download(dst any) error {
	switch b.precision {
	case PrecisionComplex64:
		data, ok := dst.([]complex64)
		if !ok {
			return ErrNotImplemented
		}
		if len(data) < b.len {
			return ErrLengthMismatch
		}
		copy(data[:b.len], b.data64)
		return nil
	case PrecisionComplex128:
		data, ok := dst.([]complex128)
		if !ok {
			return ErrNotImplemented
		}
		if len(data) < b.len {
			return ErrLengthMismatch
		}
		copy(data[:b.len], b.data128)
		return nil
	default:
		return ErrNotImplemented
	}
}

func (b *mockBuffer) Close() error {
	b.data64 = nil
	b.data128 = nil
	b.len = 0
	return nil
}

type mockStream struct{}

func (s *mockStream) Synchronize() error { return nil }
func (s *mockStream) Close() error       { return nil }

type mockPlan64 struct {
	plan *algofft.Plan[complex64]
}

func (p *mockPlan64) Len() int {
	return p.plan.Len()
}

func (p *mockPlan64) Precision() PrecisionKind {
	return PrecisionComplex64
}

func (p *mockPlan64) Forward(dst, src any) error {
	out, ok := dst.([]complex64)
	if !ok {
		return ErrNotImplemented
	}
	in, ok := src.([]complex64)
	if !ok {
		return ErrNotImplemented
	}
	return p.plan.Forward(out, in)
}

func (p *mockPlan64) Inverse(dst, src any) error {
	out, ok := dst.([]complex64)
	if !ok {
		return ErrNotImplemented
	}
	in, ok := src.([]complex64)
	if !ok {
		return ErrNotImplemented
	}
	return p.plan.Inverse(out, in)
}

func (p *mockPlan64) Close() error {
	p.plan = nil
	return nil
}

type mockPlan128 struct {
	plan *algofft.Plan[complex128]
}

func (p *mockPlan128) Len() int {
	return p.plan.Len()
}

func (p *mockPlan128) Precision() PrecisionKind {
	return PrecisionComplex128
}

func (p *mockPlan128) Forward(dst, src any) error {
	out, ok := dst.([]complex128)
	if !ok {
		return ErrNotImplemented
	}
	in, ok := src.([]complex128)
	if !ok {
		return ErrNotImplemented
	}
	return p.plan.Forward(out, in)
}

func (p *mockPlan128) Inverse(dst, src any) error {
	out, ok := dst.([]complex128)
	if !ok {
		return ErrNotImplemented
	}
	in, ok := src.([]complex128)
	if !ok {
		return ErrNotImplemented
	}
	return p.plan.Inverse(out, in)
}

func (p *mockPlan128) Close() error {
	p.plan = nil
	return nil
}
