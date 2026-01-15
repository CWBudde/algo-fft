package kernels

// forwardDIT16384SixStepComplex64 computes a 16384-point forward FFT using the
// six-step (128x128 matrix) algorithm for complex64 data.
func forwardDIT16384SixStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n = 16384
		m = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]
	for i := 0; i < n; i++ {
		work[i] = src[i]
	}

	// Step 1: Transpose 128x128.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	var rowTwiddle [128]complex64
	for k := 0; k < m; k++ {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [128]complex64

	// Step 2: Row FFTs (128 FFTs of size 128).
	for r := 0; r < m; r++ {
		row := dst[r*m : (r+1)*m]
		if !forwardDIT128Complex64(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 3: Transpose back into work.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Twiddle multiply.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			idx := i * j
			work[i*m+j] *= twiddle[idx%n]
		}
	}

	// Step 5: Row FFTs (128 FFTs of size 128).
	for r := 0; r < m; r++ {
		row := work[r*m : (r+1)*m]
		if !forwardDIT128Complex64(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose into dst.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	return true
}

// inverseDIT16384SixStepComplex64 computes a 16384-point inverse FFT using the
// six-step (128x128 matrix) algorithm for complex64 data.
func inverseDIT16384SixStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n = 16384
		m = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]
	for i := 0; i < n; i++ {
		work[i] = src[i]
	}

	// Step 1: Transpose 128x128.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	var rowTwiddle [128]complex64
	for k := 0; k < m; k++ {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [128]complex64

	// Step 2: Row IFFTs (128 IFFTs of size 128).
	for r := 0; r < m; r++ {
		row := dst[r*m : (r+1)*m]
		if !inverseDIT128Complex64(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 3: Transpose back into work.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Conjugate twiddle multiply.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			idx := i * j
			tw := twiddle[idx%n]
			work[i*m+j] *= complex(real(tw), -imag(tw))
		}
	}

	// Step 5: Row IFFTs (128 IFFTs of size 128).
	for r := 0; r < m; r++ {
		row := work[r*m : (r+1)*m]
		if !inverseDIT128Complex64(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose into dst (row IFFTs already applied 1/N scaling).
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	return true
}

// forwardDIT16384SixStepComplex128 computes a 16384-point forward FFT using the
// six-step (128x128 matrix) algorithm for complex128 data.
func forwardDIT16384SixStepComplex128(dst, src, twiddle, scratch []complex128) bool {
	const (
		n = 16384
		m = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]
	for i := 0; i < n; i++ {
		work[i] = src[i]
	}

	// Step 1: Transpose 128x128.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	var rowTwiddle [128]complex128
	for k := 0; k < m; k++ {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [128]complex128

	// Step 2: Row FFTs (128 FFTs of size 128).
	for r := 0; r < m; r++ {
		row := dst[r*m : (r+1)*m]
		if !forwardDIT128Complex128(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 3: Transpose back into work.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Twiddle multiply.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			idx := i * j
			work[i*m+j] *= twiddle[idx%n]
		}
	}

	// Step 5: Row FFTs (128 FFTs of size 128).
	for r := 0; r < m; r++ {
		row := work[r*m : (r+1)*m]
		if !forwardDIT128Complex128(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose into dst.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	return true
}

// inverseDIT16384SixStepComplex128 computes a 16384-point inverse FFT using the
// six-step (128x128 matrix) algorithm for complex128 data.
func inverseDIT16384SixStepComplex128(dst, src, twiddle, scratch []complex128) bool {
	const (
		n = 16384
		m = 128
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	work := scratch[:n]
	for i := 0; i < n; i++ {
		work[i] = src[i]
	}

	// Step 1: Transpose 128x128.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	var rowTwiddle [128]complex128
	for k := 0; k < m; k++ {
		rowTwiddle[k] = twiddle[k*m]
	}

	var rowScratch [128]complex128

	// Step 2: Row IFFTs (128 IFFTs of size 128).
	for r := 0; r < m; r++ {
		row := dst[r*m : (r+1)*m]
		if !inverseDIT128Complex128(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 3: Transpose back into work.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			work[i*m+j] = dst[j*m+i]
		}
	}

	// Step 4: Conjugate twiddle multiply.
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			idx := i * j
			tw := twiddle[idx%n]
			work[i*m+j] *= complex(real(tw), -imag(tw))
		}
	}

	// Step 5: Row IFFTs (128 IFFTs of size 128).
	for r := 0; r < m; r++ {
		row := work[r*m : (r+1)*m]
		if !inverseDIT128Complex128(row, row, rowTwiddle[:], rowScratch[:]) {
			return false
		}
	}

	// Step 6: Final transpose into dst (row IFFTs already applied 1/N scaling).
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			dst[i*m+j] = work[j*m+i]
		}
	}

	return true
}
