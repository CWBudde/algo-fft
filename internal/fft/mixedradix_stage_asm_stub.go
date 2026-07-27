//go:build !amd64 || purego

package fft

// Fused mixed-radix stage kernels exist only for AVX2. Everywhere else the
// two-pass Go stage in mixedradix_stage_twiddle.go handles every radix.

func mixedRadixStageAsm64(dst, input, table []complex64, n, span, radix int, inverse bool) bool {
	return false
}

func mixedRadixStageAsm128(dst, input, table []complex128, n, span, radix int, inverse bool) bool {
	return false
}
