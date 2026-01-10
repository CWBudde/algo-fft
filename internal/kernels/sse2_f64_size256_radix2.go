//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

func init() {
	// Size 256: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCore128(amd64.ForwardSSE2Size256Radix2Complex128Asm),
		Inverse:    wrapCore128(amd64.InverseSSE2Size256Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit256_radix2_sse2",
		Priority:   10,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}
