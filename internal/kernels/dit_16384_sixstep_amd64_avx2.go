//go:build amd64 && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// forwardDIT16384SixStepAVX2Complex64 computes a 16384-point forward FFT using the
// six-step (128×128 matrix) algorithm with AVX2-accelerated operations.
//
// This implementation uses:
// - AVX2 assembly for transpose operations (Steps 1, 6)
// - AVX2 assembly for fused transpose+twiddle (Steps 3+4)
// - Existing ForwardAVX2Size128Radix4Then2Complex64Asm kernel (radix-4-then-2) for row FFTs (Steps 2, 5).
func forwardDIT16384SixStepAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n = 16384
		m = 128 // sqrt(16384)
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Work buffer
	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-2 order)
	for i := range n {
		work[i] = src[i]
	}

	// Step 1: Transpose work -> dst (AVX2 accelerated)
	if !amd64.Transpose128x128Complex64AVX2Asm(dst, work) {
		return false
	}

	// Precompute row twiddles for size-128 FFT (stride by 128 to get W_128^k from W_16384^(k*128))
	var rowTwiddle [128]complex64
	for k := range m {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [128]complex64

	// Step 2: Row FFTs using size-128 AVX2 mixed-radix kernel (128 FFTs of size 128)
	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !amd64.ForwardAVX2Size128Radix4Then2Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Steps 3+4 fused: Transpose and twiddle multiply (AVX2 accelerated)
	// dst[i*m+j] = work[j*m+i] * W_16384^(i*j)
	if !amd64.TransposeTwiddle128x128Complex64AVX2Asm(work, dst, twiddle) {
		return false
	}

	// Step 5: Row FFTs using size-128 AVX2 mixed-radix kernel (128 FFTs of size 128)
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !amd64.ForwardAVX2Size128Radix4Then2Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose work -> dst (AVX2 accelerated)
	if !amd64.Transpose128x128Complex64AVX2Asm(dst, work) {
		return false
	}

	return true
}

// inverseDIT16384SixStepAVX2Complex64 computes a 16384-point inverse FFT using the
// six-step (128×128 matrix) algorithm with AVX2-accelerated operations.
func inverseDIT16384SixStepAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n = 16384
		m = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-2 order)
	for i := range n {
		work[i] = src[i]
	}

	// Step 1: Transpose work -> dst (AVX2 accelerated)
	if !amd64.Transpose128x128Complex64AVX2Asm(dst, work) {
		return false
	}

	// Precompute row twiddles
	var rowTwiddle [128]complex64
	for k := range m {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [128]complex64

	// Step 2: Row IFFTs using size-128 AVX2 mixed-radix kernel
	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !amd64.InverseAVX2Size128Radix4Then2Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Steps 3+4 fused: Transpose and conjugate twiddle multiply (AVX2 accelerated)
	// dst[i*m+j] = work[j*m+i] * conj(W_16384^(i*j))
	if !amd64.TransposeTwiddleConj128x128Complex64AVX2Asm(work, dst, twiddle) {
		return false
	}

	// Step 5: Row IFFTs using size-128 AVX2 mixed-radix kernel
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !amd64.InverseAVX2Size128Radix4Then2Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose work -> dst (AVX2 accelerated)
	if !amd64.Transpose128x128Complex64AVX2Asm(dst, work) {
		return false
	}

	return true
}
