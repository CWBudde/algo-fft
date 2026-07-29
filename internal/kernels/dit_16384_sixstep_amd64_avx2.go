//go:build amd64 && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// The row FFTs of the six-step decompositions run at a fixed sub-size of 128,
// so the twiddle table the size-generic radix-4 kernel wants can be prepared
// once at package load instead of gathered from the caller's length-n table on
// every transform. 128 = 2·4³, so that kernel runs its radix-4 stages to 64 and
// combines with a radix-2 tail — a radix-4-then-2 at this size by construction
// rather than by having a file named for it.
//
// The table is the packed per-stage plane layout of prepareTwiddleRadix4AVX2,
// length n+4, and it conjugates at prepare time — so forward and inverse need
// separate tables. Shared with dit_8192_sixstep_64x128_amd64_avx2.go.
//
//nolint:gochecknoglobals // twiddle tables, built once at package load
var (
	sixStepRow128FwdTwiddleC64 = newSixStepRow128TwiddleC64(false)
	sixStepRow128InvTwiddleC64 = newSixStepRow128TwiddleC64(true)
)

// sixStepRow128Size is the row-FFT length shared by the 8192 (64×128) and
// 16384 (128×128) six-step decompositions.
const sixStepRow128Size = 128

func newSixStepRow128TwiddleC64(inverse bool) []complex64 {
	table := make([]complex64, twiddleSizeRadix4AVX2(sixStepRow128Size))
	prepareTwiddleRadix4AVX2(sixStepRow128Size, inverse, table)

	return table
}

// forwardDIT16384SixStepAVX2Complex64 computes a 16384-point forward FFT using the
// six-step (128×128 matrix) algorithm with AVX2-accelerated operations.
//
// This implementation uses:
// - AVX2 assembly for transpose operations (Steps 1, 6)
// - AVX2 assembly for fused transpose+twiddle (Steps 3+4)
// - The size-generic 256-bit radix-4 kernel for the row FFTs (Steps 2, 5).
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

	var rowScratch [128]complex64

	// Step 2: Row FFTs using the size-generic radix-4 kernel (128 FFTs of size 128)
	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !forwardRadix4AVX2Complex64(row, row, sixStepRow128FwdTwiddleC64, rowScratch[:]) {
			return false
		}
	}

	// Steps 3+4 fused: Transpose and twiddle multiply (AVX2 accelerated)
	// dst[i*m+j] = work[j*m+i] * W_16384^(i*j)
	if !amd64.TransposeTwiddle128x128Complex64AVX2Asm(work, dst, twiddle) {
		return false
	}

	// Step 5: Row FFTs using the size-generic radix-4 kernel (128 FFTs of size 128)
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !forwardRadix4AVX2Complex64(row, row, sixStepRow128FwdTwiddleC64, rowScratch[:]) {
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

	var rowScratch [128]complex64

	// Step 2: Row IFFTs using the size-generic radix-4 kernel
	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !inverseRadix4AVX2Complex64(row, row, sixStepRow128InvTwiddleC64, rowScratch[:]) {
			return false
		}
	}

	// Steps 3+4 fused: Transpose and conjugate twiddle multiply (AVX2 accelerated)
	// dst[i*m+j] = work[j*m+i] * conj(W_16384^(i*j))
	if !amd64.TransposeTwiddleConj128x128Complex64AVX2Asm(work, dst, twiddle) {
		return false
	}

	// Step 5: Row IFFTs using the size-generic radix-4 kernel
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !inverseRadix4AVX2Complex64(row, row, sixStepRow128InvTwiddleC64, rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose work -> dst (AVX2 accelerated)
	if !amd64.Transpose128x128Complex64AVX2Asm(dst, work) {
		return false
	}

	return true
}
