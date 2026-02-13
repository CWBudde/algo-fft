//go:build cuda

package gpu

// CUDABackend is a stub backend enabled with the "cuda" build tag.
// It does not provide a working implementation yet.
type CUDABackend struct{}

func (b *CUDABackend) Info() BackendInfo {
	return BackendInfo{
		Name:        "cuda",
		Version:     "stub",
		Description: "CUDA backend stub (no implementation)",
	}
}

func (b *CUDABackend) Available() bool {
	return false
}

func (b *CUDABackend) Devices() ([]DeviceInfo, error) {
	return nil, ErrBackendUnavailable
}

func (b *CUDABackend) NewContext(_ int) (Context, error) {
	return nil, ErrBackendUnavailable
}

// RegisterCUDABackend registers the CUDA backend stub.
func RegisterCUDABackend() {
	RegisterBackend(&CUDABackend{})
}
