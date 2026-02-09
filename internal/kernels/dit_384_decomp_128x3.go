//go:build !amd64 || !asm || purego

package kernels

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

// forwardDIT384MixedComplex64 computes a 384-point forward FFT using the
// 128x3 decomposition (radix-3 first, then 128-point FFTs).
func forwardDIT384MixedComplex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n      = 384
		stride = 128
	)

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch[:n]

	// Step 1: Compute 128 radix-3 column DFTs.
	for n1 := range stride {
		a0 := src[n1]
		a1 := src[n1+stride]
		a2 := src[n1+2*stride]
		y0, y1, y2 := butterfly3ForwardComplex64(a0, a1, a2)
		work[n1] = y0
		work[n1+stride] = y1
		work[n1+2*stride] = y2
	}

	// Step 2: Apply twiddle factors W_384^(n1*k2).
	for n1 := range stride {
		work[stride+n1] *= twiddle[n1]
		work[2*stride+n1] *= twiddle[2*n1]
	}

	// Step 3: Compute 3 independent 128-point FFTs.
	var (
		subTwiddle [stride]complex64
		subScratch [stride]complex64
	)

	for k := range stride {
		subTwiddle[k] = twiddle[k*3]
	}

	for k2 := range 3 {
		rowStart := k2 * stride
		if !forwardDIT128Complex64(
			dst[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			subTwiddle[:],
			subScratch[:],
		) {
			return false
		}
	}

	// Step 4: Interleave output into final order.
	copy(work, dst[:n])

	for k1 := range stride {
		for k2 := range 3 {
			dst[k1*3+k2] = work[k2*stride+k1]
		}
	}

	return true
}

// inverseDIT384MixedComplex64 computes a 384-point inverse FFT.
func inverseDIT384MixedComplex64(dst, src, twiddle, scratch []complex64) bool {
	const (
		n      = 384
		stride = 128
	)

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch[:n]

	// Step 1: De-interleave input into rows.
	for k1 := range stride {
		for k2 := range 3 {
			work[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Step 2: Compute 3 independent 128-point IFFTs.
	var (
		subTwiddle [stride]complex64
		subScratch [stride]complex64
	)

	for k := range stride {
		subTwiddle[k] = twiddle[k*3]
	}

	for k2 := range 3 {
		rowStart := k2 * stride
		if !inverseDIT128Complex64(
			dst[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			subTwiddle[:],
			subScratch[:],
		) {
			return false
		}
	}

	// Step 3: Apply conjugate twiddle factors.
	copy(work, dst[:n])

	for n1 := range stride {
		work[stride+n1] *= mathpkg.Conj(twiddle[n1])
		work[2*stride+n1] *= mathpkg.Conj(twiddle[2*n1])
	}

	// Step 4: Compute 128 radix-3 inverse column butterflies.
	scale := complex64(complex(1.0/3.0, 0))

	for n1 := range stride {
		a0 := work[n1]
		a1 := work[n1+stride]
		a2 := work[n1+2*stride]
		y0, y1, y2 := butterfly3InverseComplex64(a0, a1, a2)
		dst[n1] = y0 * scale
		dst[n1+stride] = y1 * scale
		dst[n1+2*stride] = y2 * scale
	}

	return true
}

// forwardDIT384MixedComplex128 computes a 384-point forward FFT (complex128).
func forwardDIT384MixedComplex128(dst, src, twiddle, scratch []complex128) bool {
	const (
		n      = 384
		stride = 128
	)

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch[:n]

	// Step 1: Compute 128 radix-3 column DFTs.
	for n1 := range stride {
		a0 := src[n1]
		a1 := src[n1+stride]
		a2 := src[n1+2*stride]
		y0, y1, y2 := butterfly3ForwardComplex128(a0, a1, a2)
		work[n1] = y0
		work[n1+stride] = y1
		work[n1+2*stride] = y2
	}

	// Step 2: Apply twiddle factors W_384^(n1*k2).
	for n1 := range stride {
		work[stride+n1] *= twiddle[n1]
		work[2*stride+n1] *= twiddle[2*n1]
	}

	// Step 3: Compute 3 independent 128-point FFTs.
	var (
		subTwiddle [stride]complex128
		subScratch [stride]complex128
	)

	for k := range stride {
		subTwiddle[k] = twiddle[k*3]
	}

	for k2 := range 3 {
		rowStart := k2 * stride
		if !forwardDIT128Complex128(
			dst[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			subTwiddle[:],
			subScratch[:],
		) {
			return false
		}
	}

	// Step 4: Interleave output into final order.
	copy(work, dst[:n])

	for k1 := range stride {
		for k2 := range 3 {
			dst[k1*3+k2] = work[k2*stride+k1]
		}
	}

	return true
}

// inverseDIT384MixedComplex128 computes a 384-point inverse FFT (complex128).
func inverseDIT384MixedComplex128(dst, src, twiddle, scratch []complex128) bool {
	const (
		n      = 384
		stride = 128
	)

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch[:n]

	// Step 1: De-interleave input into rows.
	for k1 := range stride {
		for k2 := range 3 {
			work[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Step 2: Compute 3 independent 128-point IFFTs.
	var (
		subTwiddle [stride]complex128
		subScratch [stride]complex128
	)

	for k := range stride {
		subTwiddle[k] = twiddle[k*3]
	}

	for k2 := range 3 {
		rowStart := k2 * stride
		if !inverseDIT128Complex128(
			dst[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			subTwiddle[:],
			subScratch[:],
		) {
			return false
		}
	}

	// Step 3: Apply conjugate twiddle factors.
	copy(work, dst[:n])

	for n1 := range stride {
		work[stride+n1] *= mathpkg.Conj(twiddle[n1])
		work[2*stride+n1] *= mathpkg.Conj(twiddle[2*n1])
	}

	// Step 4: Compute 128 radix-3 inverse column butterflies.
	scale := complex128(complex(1.0/3.0, 0))

	for n1 := range stride {
		a0 := work[n1]
		a1 := work[n1+stride]
		a2 := work[n1+2*stride]
		y0, y1, y2 := butterfly3InverseComplex128(a0, a1, a2)
		dst[n1] = y0 * scale
		dst[n1+stride] = y1 * scale
		dst[n1+2*stride] = y2 * scale
	}

	return true
}
