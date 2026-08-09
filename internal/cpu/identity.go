package cpu

import (
	"fmt"
	"runtime"
	"strings"
)

// WisdomCPUIdentifier returns a stable identifier for the CPU characteristics
// that can change an empirically measured FFT winner. It includes the
// architecture, vendor/family/model when detection provides them, and the L1d
// and L2 cache geometry used by cache-sensitive plans.
//
// On targets without a model-identification source (including wasm), the
// identifier explicitly contains "unknown". Such targets remain separated by
// architecture and cache geometry, but cannot distinguish two models with the
// same reported context.
func WisdomCPUIdentifier(features Features) string {
	architecture := features.Architecture
	if architecture == "" {
		architecture = runtime.GOARCH
	}

	vendor := sanitizeIdentityComponent(features.CPUVendor)
	if vendor == "" {
		vendor = "unknown"
	}

	caches := DetectCaches()

	return fmt.Sprintf(
		"%s_%s_f%d_m%d_l1d%d_l2%d",
		sanitizeIdentityComponent(architecture),
		vendor,
		features.CPUFamily,
		features.CPUModel,
		caches.L1DataBytes,
		caches.L2Bytes,
	)
}

func sanitizeIdentityComponent(value string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return r
		default:
			return '_'
		}
	}, value)
}
