package kernels

// forwardDIT16Radix16Complex64 computes a 16-point forward FFT using a single
// radix-16 stage for complex64 data (bit-reversed input -> natural output).
func forwardDIT16Radix16Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 16
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	br := bitrevSize16Radix2
	s := src[:n]

	v0 := s[br[0]]
	v1 := s[br[1]]
	v2 := s[br[2]]
	v3 := s[br[3]]
	v4 := s[br[4]]
	v5 := s[br[5]]
	v6 := s[br[6]]
	v7 := s[br[7]]
	v8 := s[br[8]]
	v9 := s[br[9]]
	v10 := s[br[10]]
	v11 := s[br[11]]
	v12 := s[br[12]]
	v13 := s[br[13]]
	v14 := s[br[14]]
	v15 := s[br[15]]

	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex64(
		v0, v1, v2, v3, v4, v5, v6, v7,
		v8, v9, v10, v11, v12, v13, v14, v15,
	)

	dst[0] = v0
	dst[1] = v1
	dst[2] = v2
	dst[3] = v3
	dst[4] = v4
	dst[5] = v5
	dst[6] = v6
	dst[7] = v7
	dst[8] = v8
	dst[9] = v9
	dst[10] = v10
	dst[11] = v11
	dst[12] = v12
	dst[13] = v13
	dst[14] = v14
	dst[15] = v15

	return true
}

// inverseDIT16Radix16Complex64 computes a 16-point inverse FFT using a single
// radix-16 stage for complex64 data (bit-reversed input -> natural output).
func inverseDIT16Radix16Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 16
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	br := bitrevSize16Radix2
	s := src[:n]

	v0 := s[br[0]]
	v1 := s[br[1]]
	v2 := s[br[2]]
	v3 := s[br[3]]
	v4 := s[br[4]]
	v5 := s[br[5]]
	v6 := s[br[6]]
	v7 := s[br[7]]
	v8 := s[br[8]]
	v9 := s[br[9]]
	v10 := s[br[10]]
	v11 := s[br[11]]
	v12 := s[br[12]]
	v13 := s[br[13]]
	v14 := s[br[14]]
	v15 := s[br[15]]

	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex64Inverse(
		v0, v1, v2, v3, v4, v5, v6, v7,
		v8, v9, v10, v11, v12, v13, v14, v15,
	)

	const scale = 1.0 / 16.0

	dst[0] = v0 * scale
	dst[1] = v1 * scale
	dst[2] = v2 * scale
	dst[3] = v3 * scale
	dst[4] = v4 * scale
	dst[5] = v5 * scale
	dst[6] = v6 * scale
	dst[7] = v7 * scale
	dst[8] = v8 * scale
	dst[9] = v9 * scale
	dst[10] = v10 * scale
	dst[11] = v11 * scale
	dst[12] = v12 * scale
	dst[13] = v13 * scale
	dst[14] = v14 * scale
	dst[15] = v15 * scale

	return true
}

// forwardDIT16Radix16Complex128 computes a 16-point forward FFT using a single
// radix-16 stage for complex128 data (bit-reversed input -> natural output).
func forwardDIT16Radix16Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 16
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	br := bitrevSize16Radix2
	s := src[:n]

	v0 := s[br[0]]
	v1 := s[br[1]]
	v2 := s[br[2]]
	v3 := s[br[3]]
	v4 := s[br[4]]
	v5 := s[br[5]]
	v6 := s[br[6]]
	v7 := s[br[7]]
	v8 := s[br[8]]
	v9 := s[br[9]]
	v10 := s[br[10]]
	v11 := s[br[11]]
	v12 := s[br[12]]
	v13 := s[br[13]]
	v14 := s[br[14]]
	v15 := s[br[15]]

	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex128(
		v0, v1, v2, v3, v4, v5, v6, v7,
		v8, v9, v10, v11, v12, v13, v14, v15,
	)

	dst[0] = v0
	dst[1] = v1
	dst[2] = v2
	dst[3] = v3
	dst[4] = v4
	dst[5] = v5
	dst[6] = v6
	dst[7] = v7
	dst[8] = v8
	dst[9] = v9
	dst[10] = v10
	dst[11] = v11
	dst[12] = v12
	dst[13] = v13
	dst[14] = v14
	dst[15] = v15

	return true
}

// inverseDIT16Radix16Complex128 computes a 16-point inverse FFT using a single
// radix-16 stage for complex128 data (bit-reversed input -> natural output).
func inverseDIT16Radix16Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 16
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	br := bitrevSize16Radix2
	s := src[:n]

	v0 := s[br[0]]
	v1 := s[br[1]]
	v2 := s[br[2]]
	v3 := s[br[3]]
	v4 := s[br[4]]
	v5 := s[br[5]]
	v6 := s[br[6]]
	v7 := s[br[7]]
	v8 := s[br[8]]
	v9 := s[br[9]]
	v10 := s[br[10]]
	v11 := s[br[11]]
	v12 := s[br[12]]
	v13 := s[br[13]]
	v14 := s[br[14]]
	v15 := s[br[15]]

	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex128Inverse(
		v0, v1, v2, v3, v4, v5, v6, v7,
		v8, v9, v10, v11, v12, v13, v14, v15,
	)

	const scale = 1.0 / 16.0

	dst[0] = v0 * scale
	dst[1] = v1 * scale
	dst[2] = v2 * scale
	dst[3] = v3 * scale
	dst[4] = v4 * scale
	dst[5] = v5 * scale
	dst[6] = v6 * scale
	dst[7] = v7 * scale
	dst[8] = v8 * scale
	dst[9] = v9 * scale
	dst[10] = v10 * scale
	dst[11] = v11 * scale
	dst[12] = v12 * scale
	dst[13] = v13 * scale
	dst[14] = v14 * scale
	dst[15] = v15 * scale

	return true
}
