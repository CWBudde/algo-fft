//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
)

func radix5AVX2Available() bool {
	features := cpu.DetectFeatures()
	return features.HasAVX2 && !features.ForceGeneric
}

func butterfly5ForwardAVX2Complex64Slices(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64) {
	amd64.Butterfly5ForwardAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4)
}

func butterfly5InverseAVX2Complex64Slices(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64) {
	amd64.Butterfly5InverseAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4)
}
