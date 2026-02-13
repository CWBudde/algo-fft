//go:build opencl

package gpu

// OpenCLBackend is a stub backend enabled with the "opencl" build tag.
// It does not provide a working implementation yet.
type OpenCLBackend struct{}

func (b *OpenCLBackend) Info() BackendInfo {
	return BackendInfo{
		Name:        "opencl",
		Version:     "stub",
		Description: "OpenCL backend stub (no implementation)",
	}
}

func (b *OpenCLBackend) Available() bool {
	return false
}

func (b *OpenCLBackend) Devices() ([]DeviceInfo, error) {
	return nil, ErrBackendUnavailable
}

func (b *OpenCLBackend) NewContext(_ int) (Context, error) {
	return nil, ErrBackendUnavailable
}

// RegisterOpenCLBackend registers the OpenCL backend stub.
func RegisterOpenCLBackend() {
	RegisterBackend(&OpenCLBackend{})
}
