//go:build !amd64 || purego

package fft

// Fused mixed-radix stage kernels exist only for AVX2. Everywhere else the
// two-pass Go stage in mixedradix_stage_twiddle.go handles every radix.

// mixedRadixStageFused always reports false here, which is what keeps radix 7
// out of the vectorised path on these builds: its two-pass form is a measured
// regression, so admitting it without a fused kernel would cost time.
func mixedRadixStageFused(span, radix int) bool {
	return false
}

func mixedRadixStageAsm64(dst, input, table []complex64, n, span, radix int, inverse bool) bool {
	return false
}

func mixedRadixStageAsm128(dst, input, table []complex128, n, span, radix int, inverse bool) bool {
	return false
}
