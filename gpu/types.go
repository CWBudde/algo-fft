package gpu

import algofft "github.com/cwbudde/algo-fft"

// Complex is the shared complex constraint used by algofft.
type Complex = algofft.Complex

// PrecisionKind describes the precision for a GPU plan.
type PrecisionKind uint8

const (
	PrecisionComplex64 PrecisionKind = iota
	PrecisionComplex128
)

// DeviceInfo describes a GPU device.
type DeviceInfo struct {
	Name       string
	Vendor     string
	Driver     string
	MemoryMB   int
	ComputeCap string
}

// BackendInfo describes a backend implementation.
type BackendInfo struct {
	Name        string
	Version     string
	Description string
}

// PlanOptions controls GPU plan creation.
type PlanOptions struct {
	// DeviceIndex selects which device to use (0 = default).
	DeviceIndex int

	// StreamCount requests a number of execution streams/queues.
	StreamCount int

	// InPlace enables in-place transforms when supported.
	InPlace bool
}
