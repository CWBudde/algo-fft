package fft

func forwardDIT64Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex64
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !forwardDIT32Complex64(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !forwardDIT32Complex64(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		t := twiddle[k] * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	return true
}

func inverseDIT64Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex64
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !inverseDIT32Complex64(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !inverseDIT32Complex64(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		tw := twiddle[k]
		tw = complex(real(tw), -imag(tw))
		t := tw * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	scale := complex(float32(0.5), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT64Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex128
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !forwardDIT32Complex128(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !forwardDIT32Complex128(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		t := twiddle[k] * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	return true
}

func inverseDIT64Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const (
		n    = 64
		half = 32
	)

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	var bitrev32 [half]int
	for i := range half {
		bitrev32[i] = reverseBits(i, 5)
	}

	var twiddle32 [half]complex128
	for i := range half {
		twiddle32[i] = twiddle[i*2]
	}

	evenSrc := scratch[:half]

	oddSrc := scratch[half:n]
	for i := range half {
		base := i * 2
		evenSrc[i] = src[base]
		oddSrc[i] = src[base+1]
	}

	if !inverseDIT32Complex128(dst[:half], evenSrc, twiddle32[:], oddSrc, bitrev32[:]) {
		return false
	}

	if !inverseDIT32Complex128(dst[half:n], oddSrc, twiddle32[:], evenSrc, bitrev32[:]) {
		return false
	}

	for k := range half {
		tw := twiddle[k]
		tw = complex(real(tw), -imag(tw))
		t := tw * dst[half+k]
		u := dst[k]
		dst[k] = u + t
		dst[half+k] = u - t
	}

	scale := complex(0.5, 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
