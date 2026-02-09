//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

func radix3AVX2Available() bool {
	features := cpu.DetectFeatures()
	return features.HasAVX2 && !features.ForceGeneric
}

func butterfly3ForwardAVX2Complex64Slices(y0, y1, y2, a0, a1, a2 []complex64) {
	amd64.Butterfly3ForwardAVX2Complex64(y0, y1, y2, a0, a1, a2)
}

func butterfly3InverseAVX2Complex64Slices(y0, y1, y2, a0, a1, a2 []complex64) {
	amd64.Butterfly3InverseAVX2Complex64(y0, y1, y2, a0, a1, a2)
}
