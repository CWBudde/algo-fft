package gpu

import "sync"

// Backend is implemented by GPU backends (CUDA, ROCm, Metal, Vulkan, etc.).
// It is responsible for device discovery, buffer allocation, and execution.
type Backend interface {
	Info() BackendInfo
	Available() bool
	Devices() ([]DeviceInfo, error)
	NewContext(deviceIndex int) (Context, error)
}

// Context represents a backend-specific GPU context tied to a device.
type Context interface {
	Device() DeviceInfo
	// NewBuffer allocates a device buffer for complex data (interleaved real/imag).
	NewBuffer(elemCount int, precision PrecisionKind) (Buffer, error)
	// NewStream creates an execution stream/queue.
	NewStream() (Stream, error)
	// NewFFTPlan creates a backend-specific FFT plan implementation.
	NewFFTPlan(n int, precision PrecisionKind, opts PlanOptions) (PlanImpl, error)
	Close() error
}

// Buffer is a device buffer.
type Buffer interface {
	Len() int
	Precision() PrecisionKind
	// Upload copies from host to device.
	Upload(src any) error
	// Download copies from device to host.
	Download(dst any) error
	Close() error
}

// Stream represents an execution queue/stream.
type Stream interface {
	Synchronize() error
	Close() error
}

// PlanImpl is a backend-specific FFT plan implementation.
// It is intentionally untyped to avoid leaking backend-specific buffer types.
type PlanImpl interface {
	Len() int
	Precision() PrecisionKind
	Forward(dst, src any) error
	Inverse(dst, src any) error
	Close() error
}

var (
	backendMu sync.RWMutex
	backend   Backend
)

// RegisterBackend registers a GPU backend. Passing nil clears the backend.
func RegisterBackend(b Backend) {
	backendMu.Lock()
	backend = b
	backendMu.Unlock()
}

// CurrentBackendInfo reports the currently registered backend, if any.
func CurrentBackendInfo() (BackendInfo, bool) {
	backendMu.RLock()
	b := backend
	backendMu.RUnlock()
	if b == nil {
		return BackendInfo{}, false
	}
	return b.Info(), true
}

func getBackend() Backend {
	backendMu.RLock()
	b := backend
	backendMu.RUnlock()
	return b
}
