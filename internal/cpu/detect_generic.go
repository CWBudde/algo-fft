//go:build !amd64 && !arm64 && !386

package cpu

import "runtime"

// detectFeaturesImpl is the fallback implementation for architectures
// other than amd64 and arm64.
//
// This implementation returns a Features struct with all SIMD flags set to false,
// indicating that only generic (non-SIMD) kernels should be used.
// The Architecture field is set to runtime.GOARCH for informational purposes.
func detectFeaturesImpl() Features {
	return Features{
		Architecture: runtime.GOARCH,
	}
}
