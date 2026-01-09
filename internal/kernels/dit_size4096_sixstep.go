package kernels

import (
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

var bitrev4096Radix4 = mathpkg.ComputeBitReversalIndicesRadix4(4096)

// forwardDIT4096SixStepComplex64 computes a 4096-point forward FFT using the
// six-step (64×64 matrix) algorithm for complex64 data.
//
// Algorithm (Cooley-Tukey 64×64 matrix decomposition):
//  1. Transpose: View input as 64×64 matrix and transpose
//  2. Row FFTs: Apply 64-point FFT to each of 64 rows
//  3. Transpose: Transpose back
//  4. Twiddle: Multiply element (i,j) by W_4096^(i*j)
//  5. Row FFTs: Apply 64-point FFT to each of 64 rows
//  6. Transpose: Final transpose to natural order
//
// This reduces from 6 radix-4 stages to 2 sets of 3 radix-4 stages (FFT-64),
// with better cache locality due to processing contiguous rows.
func forwardDIT4096SixStepComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const (
		n = 4096
		m = 64 // sqrt(4096)
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Work buffer
	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-4 order)
	for i := range n {
		work[bitrev4096Radix4[i]] = src[bitrev[i]]
	}

	// Step 1: Transpose (work viewed as 64×64 matrix, transposed into dst)
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	// Step 2: Row FFTs (64 FFTs of size 64)
	rowTwiddle := make([]complex64, m)
	for k := range m {
		rowTwiddle[k] = twiddle[k*m] // Stride by 64 to get W_64^k from W_4096^(k*64)
	}

	rowBitrev := mathpkg.ComputeBitReversalIndicesRadix4(m)
	rowScratch := make([]complex64, m)

	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !forwardDIT64Radix4Complex64(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 3: Transpose (dst -> work viewed as transpose)
	for i := range m {
		for j := range m {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Twiddle multiply: work[i*m+j] *= W_4096^(i*j)
	for i := range m {
		for j := range m {
			idx := i * j // W_4096^(i*j), indices 0..3969
			work[i*m+j] *= twiddle[idx%n]
		}
	}

	// Step 5: Row FFTs (64 FFTs of size 64)
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !forwardDIT64Radix4Complex64(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 6: Final transpose (work -> dst)
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	return true
}

// inverseDIT4096SixStepComplex64 computes a 4096-point inverse FFT using the
// six-step (64×64 matrix) algorithm for complex64 data.
func inverseDIT4096SixStepComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const (
		n = 4096
		m = 64 // sqrt(4096)
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Work buffer
	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-4 order)
	for i := range n {
		work[bitrev4096Radix4[i]] = src[bitrev[i]]
	}

	// Step 1: Transpose (work viewed as 64×64 matrix, transposed into dst)
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	// Step 2: Row IFFTs (64 IFFTs of size 64)
	rowTwiddle := make([]complex64, m)
	for k := range m {
		rowTwiddle[k] = twiddle[k*m] // Stride by 64
	}

	rowBitrev := mathpkg.ComputeBitReversalIndicesRadix4(m)
	rowScratch := make([]complex64, m)

	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !inverseDIT64Radix4Complex64NoScale(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 3: Transpose (dst -> work viewed as transpose)
	for i := range m {
		for j := range m {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Conjugate twiddle multiply: work[i*m+j] *= conj(W_4096^(i*j))
	for i := range m {
		for j := range m {
			idx := i * j
			tw := twiddle[idx%n]
			work[i*m+j] *= complex(real(tw), -imag(tw)) // Conjugate
		}
	}

	// Step 5: Row IFFTs (64 IFFTs of size 64)
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !inverseDIT64Radix4Complex64NoScale(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 6: Final transpose (work -> dst)
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	// Apply 1/N scaling and copy back
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range n {
		dst[i] = dst[i] * scale
	}

	return true
}

// inverseDIT64Radix4Complex64NoScale performs a 64-point inverse FFT without 1/N scaling.
// This is needed for the six-step algorithm where scaling is applied once at the end.
func inverseDIT64Radix4Complex64NoScale(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 64

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 16 radix-4 butterflies with fused bit-reversal
	var stage1 [64]complex64

	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// For inverse: multiply by +i instead of -i
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 4 groups × 4 butterflies
	var stage2 [64]complex64

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			a0 := stage1[idx0]
			a1 := w1 * stage1[idx1]
			a2 := w2 * stage1[idx2]
			a3 := w3 * stage1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage2[idx0] = t0 + t2
			stage2[idx2] = t0 - t2
			stage2[idx1] = t1 + complex(-imag(t3), real(t3))
			stage2[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 3: 1 group × 16 butterflies (final)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 16 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 16
		idx2 := j + 32
		idx3 := j + 48

		a0 := stage2[idx0]
		a1 := w1 * stage2[idx1]
		a2 := w2 * stage2[idx2]
		a3 := w3 * stage2[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(-imag(t3), real(t3))
		work[idx3] = t1 + complex(imag(t3), -real(t3))
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// forwardDIT4096SixStepComplex128 computes a 4096-point forward FFT using the
// six-step (64×64 matrix) algorithm for complex128 data.
func forwardDIT4096SixStepComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const (
		n = 4096
		m = 64
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-4 order)
	for i := range n {
		work[bitrev4096Radix4[i]] = src[bitrev[i]]
	}

	// Step 1: Transpose
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	// Step 2: Row FFTs
	rowTwiddle := make([]complex128, m)
	for k := range m {
		rowTwiddle[k] = twiddle[k*m]
	}

	rowBitrev := mathpkg.ComputeBitReversalIndicesRadix4(m)
	rowScratch := make([]complex128, m)

	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !forwardDIT64Radix4Complex128(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 3: Transpose
	for i := range m {
		for j := range m {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Twiddle multiply
	for i := range m {
		for j := range m {
			idx := i * j
			work[i*m+j] *= twiddle[idx%n]
		}
	}

	// Step 5: Row FFTs
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !forwardDIT64Radix4Complex128(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 6: Final transpose
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	return true
}

// inverseDIT4096SixStepComplex128 computes a 4096-point inverse FFT using the
// six-step (64×64 matrix) algorithm for complex128 data.
func inverseDIT4096SixStepComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const (
		n = 4096
		m = 64
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := scratch[:n]

	// Step 0: Bit-reversal permutation into work (remap dynamic bitrev onto radix-4 order)
	for i := range n {
		work[bitrev4096Radix4[i]] = src[bitrev[i]]
	}

	// Step 1: Transpose
	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	// Step 2: Row IFFTs
	rowTwiddle := make([]complex128, m)
	for k := range m {
		rowTwiddle[k] = twiddle[k*m]
	}

	rowBitrev := mathpkg.ComputeBitReversalIndicesRadix4(m)
	rowScratch := make([]complex128, m)

	for r := range m {
		row := dst[r*m : (r+1)*m]
		if !inverseDIT64Radix4Complex128NoScale(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 3: Transpose
	for i := range m {
		for j := range m {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Conjugate twiddle multiply
	for i := range m {
		for j := range m {
			idx := i * j
			tw := twiddle[idx%n]
			work[i*m+j] *= complex(real(tw), -imag(tw))
		}
	}

	// Step 5: Row IFFTs
	for r := range m {
		row := work[r*m : (r+1)*m]
		if !inverseDIT64Radix4Complex128NoScale(row, row, rowTwiddle, rowScratch, rowBitrev) {
			return false
		}
	}

	// Step 6: Final transpose with scaling
	scale := complex(1.0/float64(n), 0)

	for i := range m {
		for j := range m {
			dst[i*m+j] = work[j*m+i]
		}
	}

	for i := range n {
		dst[i] = dst[i] * scale
	}

	return true
}

// inverseDIT64Radix4Complex128NoScale performs a 64-point inverse FFT without 1/N scaling.
func inverseDIT64Radix4Complex128NoScale(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 64

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	var stage1 [64]complex128

	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	var stage2 [64]complex128

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			a0 := stage1[idx0]
			a1 := w1 * stage1[idx1]
			a2 := w2 * stage1[idx2]
			a3 := w3 * stage1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage2[idx0] = t0 + t2
			stage2[idx2] = t0 - t2
			stage2[idx1] = t1 + complex(-imag(t3), real(t3))
			stage2[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 16 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 16
		idx2 := j + 32
		idx3 := j + 48

		a0 := stage2[idx0]
		a1 := w1 * stage2[idx1]
		a2 := w2 * stage2[idx2]
		a3 := w3 * stage2[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(-imag(t3), real(t3))
		work[idx3] = t1 + complex(imag(t3), -real(t3))
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}
