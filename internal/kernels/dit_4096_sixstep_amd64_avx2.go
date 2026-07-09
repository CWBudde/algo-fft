//go:build amd64 && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// forwardDIT4096SixStepAVX2Complex64 computes a 4096-point forward FFT using the
// six-step (64×64 matrix) algorithm with AVX2-accelerated operations.
//
// This implementation uses:
// - AVX2 assembly for transpose operations (Steps 1, 6)
// - AVX2 assembly for fused transpose+twiddle (Steps 3+4)
// - Existing ForwardAVX2Size64Radix4Complex64Asm kernel for row FFTs (Steps 2, 5).
func forwardDIT4096SixStepAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n = 4096
		m = 64 // sqrt(4096)
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Work buffer
	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-4 order)
	for i := range n {
		work[i] = src[i]
	}

	// Step 1: Transpose work → dst (AVX2 accelerated)
	if !amd64.Transpose64x64Complex64AVX2Asm(dst, work) {
		return false
	}

	// Precompute row twiddles for size-64 FFT (stride by 64 to get W_64^k from W_4096^(k*64))
	var rowTwiddle [64]complex64
	for k := range m {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [64]complex64

	// Step 2: Row FFTs using AVX2 (64 FFTs of size 64)
	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !amd64.ForwardAVX2Size64Radix4Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Steps 3+4 fused: Transpose and twiddle multiply (AVX2 accelerated)
	// dst[i*m+j] = work[j*m+i] * W_4096^(i*j)
	if !amd64.TransposeTwiddle64x64Complex64AVX2Asm(work, dst, twiddle) {
		return false
	}

	// Step 5: Row FFTs using AVX2 (64 FFTs of size 64)
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !amd64.ForwardAVX2Size64Radix4Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose work → dst (AVX2 accelerated)
	if !amd64.Transpose64x64Complex64AVX2Asm(dst, work) {
		return false
	}

	return true
}

// inverseDIT4096SixStepAVX2Complex64 computes a 4096-point inverse FFT using the
// six-step (64×64 matrix) algorithm with AVX2-accelerated operations.
func inverseDIT4096SixStepAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n = 4096
		m = 64
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-4 order)
	for i := range n {
		work[i] = src[i]
	}

	// Step 1: Transpose work → dst (AVX2 accelerated)
	if !amd64.Transpose64x64Complex64AVX2Asm(dst, work) {
		return false
	}

	// Precompute row twiddles
	var rowTwiddle [64]complex64
	for k := range m {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [64]complex64

	// Step 2: Row IFFTs using AVX2
	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !amd64.InverseAVX2Size64Radix4Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Steps 3+4 fused: Transpose and conjugate twiddle multiply (AVX2 accelerated)
	// dst[i*m+j] = work[j*m+i] * conj(W_4096^(i*j))
	if !amd64.TransposeTwiddleConj64x64Complex64AVX2Asm(work, dst, twiddle) {
		return false
	}

	// Step 5: Row IFFTs using AVX2
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !amd64.InverseAVX2Size64Radix4Complex64Asm(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose work → dst (AVX2 accelerated)
	if !amd64.Transpose64x64Complex64AVX2Asm(dst, work) {
		return false
	}

	return true
}
