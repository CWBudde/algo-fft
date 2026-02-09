//go:build amd64 && asm && !purego

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// forwardDIT8192SixStep64x128AVX2Complex64 computes an 8192-point forward FFT using
// a 64×128 matrix decomposition with AVX2-accelerated row FFTs.
//
// The algorithm decomposes 8192 = 64 × 128:
//
//	Step 1: Transpose 64×128 → 128×64
//	Step 2: 128 parallel FFT-64 (AVX2)
//	Step 3: Transpose 128×64 → 64×128 with twiddle multiply
//	Step 4: 64 parallel FFT-128 (AVX2)
//	Step 5: Final transpose 64×128 → 128×64
//
// This version uses AVX2 assembly for the row FFTs but Go for transpose.
func forwardDIT8192SixStep64x128AVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n    = 8192
		rows = 64
		cols = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]
	var work2 []complex64
	if len(scratch) >= 2*n {
		work2 = scratch[n : 2*n]
	} else {
		work2 = make([]complex64, n)
	}

	// Step 1: Transpose 64×128 (src) → 128×64 (work)
	for i := range rows {
		for j := range cols {
			work[j*rows+i] = src[i*cols+j]
		}
	}

	// Precompute row twiddles for size-64 FFT
	var rowTwiddle64 [64]complex64
	for k := range 64 {
		rowTwiddle64[k] = twiddle[k*cols]
	}

	var rowScratch64 [64]complex64

	// Step 2: 128 parallel FFT-64 using AVX2
	for r := range cols {
		row := work[r*rows : (r+1)*rows]
		if !amd64.ForwardAVX2Size64Radix4Complex64Asm(row, row, rowTwiddle64[:], rowScratch64[:]) {
			return false
		}
	}

	// Step 3: Transpose 128×64 → 64×128 with twiddle multiply
	for i := range rows {
		for j := range cols {
			tw := twiddle[(i*j)%n]
			work2[i*cols+j] = work[j*rows+i] * tw
		}
	}

	// Precompute row twiddles for size-128 FFT
	var rowTwiddle128 [128]complex64
	for k := range 128 {
		rowTwiddle128[k] = twiddle[k*rows]
	}

	var rowScratch128 [128]complex64

	// Step 4: 64 parallel FFT-128 using AVX2
	for r := range rows {
		row := work2[r*cols : (r+1)*cols]
		if !amd64.ForwardAVX2Size128Radix4Then2Complex64Asm(row, row, rowTwiddle128[:], rowScratch128[:]) {
			return false
		}
	}

	// Step 5: Final transpose 64×128 → 128×64
	for i := range rows {
		for j := range cols {
			dst[j*rows+i] = work2[i*cols+j]
		}
	}

	return true
}

// inverseDIT8192SixStep64x128AVX2Complex64 computes an 8192-point inverse FFT using
// a 64×128 matrix decomposition with AVX2-accelerated row FFTs.
func inverseDIT8192SixStep64x128AVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n    = 8192
		rows = 64
		cols = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]
	var work2 []complex64
	if len(scratch) >= 2*n {
		work2 = scratch[n : 2*n]
	} else {
		work2 = make([]complex64, n)
	}

	// Step 1: Transpose 64×128 (src) → 128×64 (work)
	for i := range rows {
		for j := range cols {
			work[j*rows+i] = src[i*cols+j]
		}
	}

	// Precompute row twiddles for size-64 IFFT
	var rowTwiddle64 [64]complex64
	for k := range 64 {
		rowTwiddle64[k] = twiddle[k*cols]
	}

	var rowScratch64 [64]complex64

	// Step 2: 128 parallel IFFT-64 using AVX2
	for r := range cols {
		row := work[r*rows : (r+1)*rows]
		if !amd64.InverseAVX2Size64Radix4Complex64Asm(row, row, rowTwiddle64[:], rowScratch64[:]) {
			return false
		}
	}

	// Step 3: Transpose 128×64 → 64×128 with conjugate twiddle multiply
	for i := range rows {
		for j := range cols {
			tw := twiddle[(i*j)%n]
			twConj := complex(real(tw), -imag(tw))
			work2[i*cols+j] = work[j*rows+i] * twConj
		}
	}

	// Precompute row twiddles for size-128 IFFT
	var rowTwiddle128 [128]complex64
	for k := range 128 {
		rowTwiddle128[k] = twiddle[k*rows]
	}

	var rowScratch128 [128]complex64

	// Step 4: 64 parallel IFFT-128 using AVX2
	for r := range rows {
		row := work2[r*cols : (r+1)*cols]
		if !amd64.InverseAVX2Size128Radix4Then2Complex64Asm(row, row, rowTwiddle128[:], rowScratch128[:]) {
			return false
		}
	}

	// Step 5: Final transpose 64×128 → 128×64
	for i := range rows {
		for j := range cols {
			dst[j*rows+i] = work2[i*cols+j]
		}
	}

	// Note: AVX2 size-128 inverse already applies 1/128 scaling
	// AVX2 size-64 inverse already applies 1/64 scaling
	// Total: 1/64 * 1/128 = 1/8192 ✓

	return true
}
