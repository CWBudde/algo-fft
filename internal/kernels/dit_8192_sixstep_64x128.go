package kernels

// forwardDIT8192SixStep64x128Complex64 computes an 8192-point forward FFT using
// a 64×128 matrix decomposition (six-step algorithm).
//
// The algorithm decomposes 8192 = 64 × 128 and computes:
//
//	Step 1: Transpose 64×128 → 128×64
//	Step 2: 128 parallel FFT-64 (on rows)
//	Step 3: Transpose 128×64 → 64×128 with twiddle multiply
//	Step 4: 64 parallel FFT-128 (on rows)
//	Step 5: Final transpose 64×128 → 128×64
//
// This reduces from 7 mixed-radix stages to 2 composite FFT stages with
// better cache behavior due to row-oriented processing.
func forwardDIT8192SixStep64x128Complex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n    = 8192
		rows = 64  // first dimension
		cols = 128 // second dimension
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Work buffer for intermediate results
	work := scratch[:n]

	var work2 []complex64
	if len(scratch) >= 2*n {
		work2 = scratch[n : 2*n]
	} else {
		work2 = make([]complex64, n) // fallback when scratch is undersized
	}

	// Step 1: Transpose 64×128 (src) → 128×64 (work)
	// src[i*cols + j] → work[j*rows + i]
	for i := range rows {
		for j := range cols {
			work[j*rows+i] = src[i*cols+j]
		}
	}

	// Precompute row twiddles for size-64 FFT
	// W_64^k = W_8192^(k*128) since 8192/64 = 128
	var rowTwiddle64 [64]complex64
	for k := range 64 {
		rowTwiddle64[k] = twiddle[k*cols]
	}

	var rowScratch64 [64]complex64

	// Step 2: 128 parallel FFT-64 (on rows of 128×64 matrix)
	for r := range cols {
		row := work[r*rows : (r+1)*rows]
		if !forwardDIT64Radix4Complex64(row, row, rowTwiddle64[:], rowScratch64[:]) {
			return false
		}
	}

	// Step 3: Transpose 128×64 → 64×128 with twiddle multiply
	// work[j*rows + i] → work2[i*cols + j] * W_8192^(i*j)
	for i := range rows {
		for j := range cols {
			tw := twiddle[(i*j)%n]
			work2[i*cols+j] = work[j*rows+i] * tw
		}
	}

	// Precompute row twiddles for size-128 FFT
	// W_128^k = W_8192^(k*64) since 8192/128 = 64
	var rowTwiddle128 [128]complex64
	for k := range 128 {
		rowTwiddle128[k] = twiddle[k*rows]
	}

	var rowScratch128 [128]complex64

	// Step 4: 64 parallel FFT-128 (on rows of 64×128 matrix)
	for r := range rows {
		row := work2[r*cols : (r+1)*cols]
		if !forwardDIT128Complex64(row, row, rowTwiddle128[:], rowScratch128[:]) {
			return false
		}
	}

	// Step 5: Final transpose 64×128 → 128×64 (to standard FFT output order)
	// work2[i*cols + j] → dst[j*rows + i]
	for i := range rows {
		for j := range cols {
			dst[j*rows+i] = work2[i*cols+j]
		}
	}

	return true
}

// inverseDIT8192SixStep64x128Complex64 computes an 8192-point inverse FFT using
// a 64×128 matrix decomposition (six-step algorithm).
//
// Uses conjugated twiddle factors. The 1/N scaling is handled by the row IFFTs.
func inverseDIT8192SixStep64x128Complex64(dst, src, twiddle, scratch []complex64) bool {
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

	// Step 2: 128 parallel IFFT-64 (on rows)
	for r := range cols {
		row := work[r*rows : (r+1)*rows]
		if !inverseDIT64Radix4Complex64(row, row, rowTwiddle64[:], rowScratch64[:]) {
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

	// Step 4: 64 parallel IFFT-128 (on rows)
	for r := range rows {
		row := work2[r*cols : (r+1)*cols]
		if !inverseDIT128Complex64(row, row, rowTwiddle128[:], rowScratch128[:]) {
			return false
		}
	}

	// Step 5: Final transpose 64×128 → 128×64
	for i := range rows {
		for j := range cols {
			dst[j*rows+i] = work2[i*cols+j]
		}
	}

	// The row IFFTs already applied 1/64 and 1/128 scaling
	// Total scaling: 1/64 * 1/128 = 1/8192 ✓

	return true
}

// forwardDIT8192SixStep64x128Complex128 computes an 8192-point forward FFT using
// a 64×128 matrix decomposition for complex128 data.
func forwardDIT8192SixStep64x128Complex128(dst, src, twiddle, scratch []complex128) bool {
	const (
		n    = 8192
		rows = 64
		cols = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]

	var work2 []complex128
	if len(scratch) >= 2*n {
		work2 = scratch[n : 2*n]
	} else {
		work2 = make([]complex128, n)
	}

	// Step 1: Transpose 64×128 → 128×64
	for i := range rows {
		for j := range cols {
			work[j*rows+i] = src[i*cols+j]
		}
	}

	// Precompute row twiddles for size-64 FFT
	var rowTwiddle64 [64]complex128
	for k := range 64 {
		rowTwiddle64[k] = twiddle[k*cols]
	}

	var rowScratch64 [64]complex128

	// Step 2: 128 parallel FFT-64
	for r := range cols {
		row := work[r*rows : (r+1)*rows]
		if !forwardDIT64Radix4Complex128(row, row, rowTwiddle64[:], rowScratch64[:]) {
			return false
		}
	}

	// Step 3: Transpose + twiddle
	for i := range rows {
		for j := range cols {
			tw := twiddle[(i*j)%n]
			work2[i*cols+j] = work[j*rows+i] * tw
		}
	}

	// Precompute row twiddles for size-128 FFT
	var rowTwiddle128 [128]complex128
	for k := range 128 {
		rowTwiddle128[k] = twiddle[k*rows]
	}

	var rowScratch128 [128]complex128

	// Step 4: 64 parallel FFT-128
	for r := range rows {
		row := work2[r*cols : (r+1)*cols]
		if !forwardDIT128Complex128(row, row, rowTwiddle128[:], rowScratch128[:]) {
			return false
		}
	}

	// Step 5: Final transpose
	for i := range rows {
		for j := range cols {
			dst[j*rows+i] = work2[i*cols+j]
		}
	}

	return true
}

// inverseDIT8192SixStep64x128Complex128 computes an 8192-point inverse FFT using
// a 64×128 matrix decomposition for complex128 data.
func inverseDIT8192SixStep64x128Complex128(dst, src, twiddle, scratch []complex128) bool {
	const (
		n    = 8192
		rows = 64
		cols = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]

	var work2 []complex128
	if len(scratch) >= 2*n {
		work2 = scratch[n : 2*n]
	} else {
		work2 = make([]complex128, n)
	}

	// Step 1: Transpose 64×128 → 128×64
	for i := range rows {
		for j := range cols {
			work[j*rows+i] = src[i*cols+j]
		}
	}

	// Precompute row twiddles for size-64 IFFT
	var rowTwiddle64 [64]complex128
	for k := range 64 {
		rowTwiddle64[k] = twiddle[k*cols]
	}

	var rowScratch64 [64]complex128

	// Step 2: 128 parallel IFFT-64
	for r := range cols {
		row := work[r*rows : (r+1)*rows]
		if !inverseDIT64Radix4Complex128(row, row, rowTwiddle64[:], rowScratch64[:]) {
			return false
		}
	}

	// Step 3: Transpose + conjugate twiddle
	for i := range rows {
		for j := range cols {
			tw := twiddle[(i*j)%n]
			twConj := complex(real(tw), -imag(tw))
			work2[i*cols+j] = work[j*rows+i] * twConj
		}
	}

	// Precompute row twiddles for size-128 IFFT
	var rowTwiddle128 [128]complex128
	for k := range 128 {
		rowTwiddle128[k] = twiddle[k*rows]
	}

	var rowScratch128 [128]complex128

	// Step 4: 64 parallel IFFT-128
	for r := range rows {
		row := work2[r*cols : (r+1)*cols]
		if !inverseDIT128Complex128(row, row, rowTwiddle128[:], rowScratch128[:]) {
			return false
		}
	}

	// Step 5: Final transpose
	for i := range rows {
		for j := range cols {
			dst[j*rows+i] = work2[i*cols+j]
		}
	}

	return true
}
