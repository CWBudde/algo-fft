package kernels

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

// bitrev32Radix2 pre-computes bit-reversal indices for a 32-point radix-2 FFT.
//
//nolint:gochecknoglobals
var bitrev32Radix2 = mathpkg.ComputeBitReversalIndices(32)

// forwardDIT32Complex64 computes a 32-point radix-2 forward FFT using the
// Decimation-in-Time (DIT) Cooley-Tukey algorithm for complex64 data.
// This implementation is fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT32Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 32

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev32Radix2
	s := src[:n]
	w1, w2, w3, w4, w5, w6, w7, w8 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7], twiddle[8]
	w9, w10, w11, w12, w13, w14, w15 := twiddle[9], twiddle[10], twiddle[11], twiddle[12], twiddle[13], twiddle[14], twiddle[15]

	// Stage 1: 16 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1
	x0 = s[br[16]]
	x1 = s[br[17]]
	a16, a17 := x0+x1, x0-x1
	x0 = s[br[18]]
	x1 = s[br[19]]
	a18, a19 := x0+x1, x0-x1
	x0 = s[br[20]]
	x1 = s[br[21]]
	a20, a21 := x0+x1, x0-x1
	x0 = s[br[22]]
	x1 = s[br[23]]
	a22, a23 := x0+x1, x0-x1
	x0 = s[br[24]]
	x1 = s[br[25]]
	a24, a25 := x0+x1, x0-x1
	x0 = s[br[26]]
	x1 = s[br[27]]
	a26, a27 := x0+x1, x0-x1
	x0 = s[br[28]]
	x1 = s[br[29]]
	a28, a29 := x0+x1, x0-x1
	x0 = s[br[30]]
	x1 = s[br[31]]
	a30, a31 := x0+x1, x0-x1

	// Stage 2: 8 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w8 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w8 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w8 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w8 * a15
	b13, b15 := a13+t, a13-t
	b16, b18 := a16+a18, a16-a18
	t = w8 * a19
	b17, b19 := a17+t, a17-t
	b20, b22 := a20+a22, a20-a22
	t = w8 * a23
	b21, b23 := a21+t, a21-t
	b24, b26 := a24+a26, a24-a26
	t = w8 * a27
	b25, b27 := a25+t, a25-t
	b28, b30 := a28+a30, a28-a30
	t = w8 * a31
	b29, b31 := a29+t, a29-t

	// Stage 3: 4 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w4 * b5
	c1, c5 := b1+t, b1-t
	t = w8 * b6
	c2, c6 := b2+t, b2-t
	t = w12 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w4 * b13
	c9, c13 := b9+t, b9-t
	t = w8 * b14
	c10, c14 := b10+t, b10-t
	t = w12 * b15
	c11, c15 := b11+t, b11-t
	c16, c20 := b16+b20, b16-b20
	t = w4 * b21
	c17, c21 := b17+t, b17-t
	t = w8 * b22
	c18, c22 := b18+t, b18-t
	t = w12 * b23
	c19, c23 := b19+t, b19-t
	c24, c28 := b24+b28, b24-b28
	t = w4 * b29
	c25, c29 := b25+t, b25-t
	t = w8 * b30
	c26, c30 := b26+t, b26-t
	t = w12 * b31
	c27, c31 := b27+t, b27-t

	// Stage 4: 2 radix-2 butterflies, stride=16
	d0, d8 := c0+c8, c0-c8
	t = w2 * c9
	d1, d9 := c1+t, c1-t
	t = w4 * c10
	d2, d10 := c2+t, c2-t
	t = w6 * c11
	d3, d11 := c3+t, c3-t
	t = w8 * c12
	d4, d12 := c4+t, c4-t
	t = w10 * c13
	d5, d13 := c5+t, c5-t
	t = w12 * c14
	d6, d14 := c6+t, c6-t
	t = w14 * c15
	d7, d15 := c7+t, c7-t
	d16, d24 := c16+c24, c16-c24
	t = w2 * c25
	d17, d25 := c17+t, c17-t
	t = w4 * c26
	d18, d26 := c18+t, c18-t
	t = w6 * c27
	d19, d27 := c19+t, c19-t
	t = w8 * c28
	d20, d28 := c20+t, c20-t
	t = w10 * c29
	d21, d29 := c21+t, c21-t
	t = w12 * c30
	d22, d30 := c22+t, c22-t
	t = w14 * c31
	d23, d31 := c23+t, c23-t

	// Stage 5: 1 radix-2 butterfly, stride=32 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[16] = d0+d16, d0-d16
	t = w1 * d17
	work[1], work[17] = d1+t, d1-t
	t = w2 * d18
	work[2], work[18] = d2+t, d2-t
	t = w3 * d19
	work[3], work[19] = d3+t, d3-t
	t = w4 * d20
	work[4], work[20] = d4+t, d4-t
	t = w5 * d21
	work[5], work[21] = d5+t, d5-t
	t = w6 * d22
	work[6], work[22] = d6+t, d6-t
	t = w7 * d23
	work[7], work[23] = d7+t, d7-t
	t = w8 * d24
	work[8], work[24] = d8+t, d8-t
	t = w9 * d25
	work[9], work[25] = d9+t, d9-t
	t = w10 * d26
	work[10], work[26] = d10+t, d10-t
	t = w11 * d27
	work[11], work[27] = d11+t, d11-t
	t = w12 * d28
	work[12], work[28] = d12+t, d12-t
	t = w13 * d29
	work[13], work[29] = d13+t, d13-t
	t = w14 * d30
	work[14], work[30] = d14+t, d14-t
	t = w15 * d31
	work[15], work[31] = d15+t, d15-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT32Complex64 computes a 32-point radix-2 inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
//
//nolint:funlen
func inverseDIT32Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 32

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev32Radix2
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3, w4, w5, w6, w7, w8 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7], twiddle[8]
	w9, w10, w11, w12, w13, w14, w15 := twiddle[9], twiddle[10], twiddle[11], twiddle[12], twiddle[13], twiddle[14], twiddle[15]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))
	w4 = complex(real(w4), -imag(w4))
	w5 = complex(real(w5), -imag(w5))
	w6 = complex(real(w6), -imag(w6))
	w7 = complex(real(w7), -imag(w7))
	w8 = complex(real(w8), -imag(w8))
	w9 = complex(real(w9), -imag(w9))
	w10 = complex(real(w10), -imag(w10))
	w11 = complex(real(w11), -imag(w11))
	w12 = complex(real(w12), -imag(w12))
	w13 = complex(real(w13), -imag(w13))
	w14 = complex(real(w14), -imag(w14))
	w15 = complex(real(w15), -imag(w15))

	// Stage 1: 16 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1

	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1

	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1

	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1

	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1

	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1

	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1

	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1

	x0 = s[br[16]]
	x1 = s[br[17]]
	a16, a17 := x0+x1, x0-x1

	x0 = s[br[18]]
	x1 = s[br[19]]
	a18, a19 := x0+x1, x0-x1

	x0 = s[br[20]]
	x1 = s[br[21]]
	a20, a21 := x0+x1, x0-x1

	x0 = s[br[22]]
	x1 = s[br[23]]
	a22, a23 := x0+x1, x0-x1

	x0 = s[br[24]]
	x1 = s[br[25]]
	a24, a25 := x0+x1, x0-x1

	x0 = s[br[26]]
	x1 = s[br[27]]
	a26, a27 := x0+x1, x0-x1

	x0 = s[br[28]]
	x1 = s[br[29]]
	a28, a29 := x0+x1, x0-x1

	x0 = s[br[30]]
	x1 = s[br[31]]
	a30, a31 := x0+x1, x0-x1

	// Stage 2: 8 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w8 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w8 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w8 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w8 * a15
	b13, b15 := a13+t, a13-t
	b16, b18 := a16+a18, a16-a18
	t = w8 * a19
	b17, b19 := a17+t, a17-t
	b20, b22 := a20+a22, a20-a22
	t = w8 * a23
	b21, b23 := a21+t, a21-t
	b24, b26 := a24+a26, a24-a26
	t = w8 * a27
	b25, b27 := a25+t, a25-t
	b28, b30 := a28+a30, a28-a30
	t = w8 * a31
	b29, b31 := a29+t, a29-t

	// Stage 3: 4 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w4 * b5
	c1, c5 := b1+t, b1-t
	t = w8 * b6
	c2, c6 := b2+t, b2-t
	t = w12 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w4 * b13
	c9, c13 := b9+t, b9-t
	t = w8 * b14
	c10, c14 := b10+t, b10-t
	t = w12 * b15
	c11, c15 := b11+t, b11-t
	c16, c20 := b16+b20, b16-b20
	t = w4 * b21
	c17, c21 := b17+t, b17-t
	t = w8 * b22
	c18, c22 := b18+t, b18-t
	t = w12 * b23
	c19, c23 := b19+t, b19-t
	c24, c28 := b24+b28, b24-b28
	t = w4 * b29
	c25, c29 := b25+t, b25-t
	t = w8 * b30
	c26, c30 := b26+t, b26-t
	t = w12 * b31
	c27, c31 := b27+t, b27-t

	// Stage 4: 2 radix-2 butterflies, stride=16
	d0, d8 := c0+c8, c0-c8
	t = w2 * c9
	d1, d9 := c1+t, c1-t
	t = w4 * c10
	d2, d10 := c2+t, c2-t
	t = w6 * c11
	d3, d11 := c3+t, c3-t
	t = w8 * c12
	d4, d12 := c4+t, c4-t
	t = w10 * c13
	d5, d13 := c5+t, c5-t
	t = w12 * c14
	d6, d14 := c6+t, c6-t
	t = w14 * c15
	d7, d15 := c7+t, c7-t
	d16, d24 := c16+c24, c16-c24
	t = w2 * c25
	d17, d25 := c17+t, c17-t
	t = w4 * c26
	d18, d26 := c18+t, c18-t
	t = w6 * c27
	d19, d27 := c19+t, c19-t
	t = w8 * c28
	d20, d28 := c20+t, c20-t
	t = w10 * c29
	d21, d29 := c21+t, c21-t
	t = w12 * c30
	d22, d30 := c22+t, c22-t
	t = w14 * c31
	d23, d31 := c23+t, c23-t

	// Stage 5: 1 radix-2 butterfly, stride=32 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[16] = d0+d16, d0-d16
	t = w1 * d17
	work[1], work[17] = d1+t, d1-t
	t = w2 * d18
	work[2], work[18] = d2+t, d2-t
	t = w3 * d19
	work[3], work[19] = d3+t, d3-t
	t = w4 * d20
	work[4], work[20] = d4+t, d4-t
	t = w5 * d21
	work[5], work[21] = d5+t, d5-t
	t = w6 * d22
	work[6], work[22] = d6+t, d6-t
	t = w7 * d23
	work[7], work[23] = d7+t, d7-t
	t = w8 * d24
	work[8], work[24] = d8+t, d8-t
	t = w9 * d25
	work[9], work[25] = d9+t, d9-t
	t = w10 * d26
	work[10], work[26] = d10+t, d10-t
	t = w11 * d27
	work[11], work[27] = d11+t, d11-t
	t = w12 * d28
	work[12], work[28] = d12+t, d12-t
	t = w13 * d29
	work[13], work[29] = d13+t, d13-t
	t = w14 * d30
	work[14], work[30] = d14+t, d14-t
	t = w15 * d31
	work[15], work[31] = d15+t, d15-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT32Complex128 computes a 32-point radix-2 forward FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT32Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 32

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev32Radix2
	s := src[:n]

	// Pre-load twiddle factors
	w1, w2, w3, w4, w5, w6, w7, w8 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7], twiddle[8]
	w9, w10, w11, w12, w13, w14, w15 := twiddle[9], twiddle[10], twiddle[11], twiddle[12], twiddle[13], twiddle[14], twiddle[15]

	// Stage 1: 16 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1
	x0 = s[br[16]]
	x1 = s[br[17]]
	a16, a17 := x0+x1, x0-x1
	x0 = s[br[18]]
	x1 = s[br[19]]
	a18, a19 := x0+x1, x0-x1
	x0 = s[br[20]]
	x1 = s[br[21]]
	a20, a21 := x0+x1, x0-x1
	x0 = s[br[22]]
	x1 = s[br[23]]
	a22, a23 := x0+x1, x0-x1
	x0 = s[br[24]]
	x1 = s[br[25]]
	a24, a25 := x0+x1, x0-x1
	x0 = s[br[26]]
	x1 = s[br[27]]
	a26, a27 := x0+x1, x0-x1
	x0 = s[br[28]]
	x1 = s[br[29]]
	a28, a29 := x0+x1, x0-x1
	x0 = s[br[30]]
	x1 = s[br[31]]
	a30, a31 := x0+x1, x0-x1

	// Stage 2: 8 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w8 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w8 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w8 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w8 * a15
	b13, b15 := a13+t, a13-t
	b16, b18 := a16+a18, a16-a18
	t = w8 * a19
	b17, b19 := a17+t, a17-t
	b20, b22 := a20+a22, a20-a22
	t = w8 * a23
	b21, b23 := a21+t, a21-t
	b24, b26 := a24+a26, a24-a26
	t = w8 * a27
	b25, b27 := a25+t, a25-t
	b28, b30 := a28+a30, a28-a30
	t = w8 * a31
	b29, b31 := a29+t, a29-t

	// Stage 3: 4 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w4 * b5
	c1, c5 := b1+t, b1-t
	t = w8 * b6
	c2, c6 := b2+t, b2-t
	t = w12 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w4 * b13
	c9, c13 := b9+t, b9-t
	t = w8 * b14
	c10, c14 := b10+t, b10-t
	t = w12 * b15
	c11, c15 := b11+t, b11-t
	c16, c20 := b16+b20, b16-b20
	t = w4 * b21
	c17, c21 := b17+t, b17-t
	t = w8 * b22
	c18, c22 := b18+t, b18-t
	t = w12 * b23
	c19, c23 := b19+t, b19-t
	c24, c28 := b24+b28, b24-b28
	t = w4 * b29
	c25, c29 := b25+t, b25-t
	t = w8 * b30
	c26, c30 := b26+t, b26-t
	t = w12 * b31
	c27, c31 := b27+t, b27-t

	// Stage 4: 2 radix-2 butterflies, stride=16
	d0, d8 := c0+c8, c0-c8
	t = w2 * c9
	d1, d9 := c1+t, c1-t
	t = w4 * c10
	d2, d10 := c2+t, c2-t
	t = w6 * c11
	d3, d11 := c3+t, c3-t
	t = w8 * c12
	d4, d12 := c4+t, c4-t
	t = w10 * c13
	d5, d13 := c5+t, c5-t
	t = w12 * c14
	d6, d14 := c6+t, c6-t
	t = w14 * c15
	d7, d15 := c7+t, c7-t
	d16, d24 := c16+c24, c16-c24
	t = w2 * c25
	d17, d25 := c17+t, c17-t
	t = w4 * c26
	d18, d26 := c18+t, c18-t
	t = w6 * c27
	d19, d27 := c19+t, c19-t
	t = w8 * c28
	d20, d28 := c20+t, c20-t
	t = w10 * c29
	d21, d29 := c21+t, c21-t
	t = w12 * c30
	d22, d30 := c22+t, c22-t
	t = w14 * c31
	d23, d31 := c23+t, c23-t

	// Stage 5: 1 radix-2 butterfly, stride=32 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[16] = d0+d16, d0-d16
	t = w1 * d17
	work[1], work[17] = d1+t, d1-t
	t = w2 * d18
	work[2], work[18] = d2+t, d2-t
	t = w3 * d19
	work[3], work[19] = d3+t, d3-t
	t = w4 * d20
	work[4], work[20] = d4+t, d4-t
	t = w5 * d21
	work[5], work[21] = d5+t, d5-t
	t = w6 * d22
	work[6], work[22] = d6+t, d6-t
	t = w7 * d23
	work[7], work[23] = d7+t, d7-t
	t = w8 * d24
	work[8], work[24] = d8+t, d8-t
	t = w9 * d25
	work[9], work[25] = d9+t, d9-t
	t = w10 * d26
	work[10], work[26] = d10+t, d10-t
	t = w11 * d27
	work[11], work[27] = d11+t, d11-t
	t = w12 * d28
	work[12], work[28] = d12+t, d12-t
	t = w13 * d29
	work[13], work[29] = d13+t, d13-t
	t = w14 * d30
	work[14], work[30] = d14+t, d14-t
	t = w15 * d31
	work[15], work[31] = d15+t, d15-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT32Complex128 computes a 32-point radix-2 inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
//
//nolint:funlen
func inverseDIT32Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 32

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hint for compiler optimization
	br := bitrev32Radix2
	s := src[:n]

	// Conjugate twiddles for inverse transform
	w1, w2, w3, w4, w5, w6, w7, w8 := twiddle[1], twiddle[2], twiddle[3], twiddle[4], twiddle[5], twiddle[6], twiddle[7], twiddle[8]
	w9, w10, w11, w12, w13, w14, w15 := twiddle[9], twiddle[10], twiddle[11], twiddle[12], twiddle[13], twiddle[14], twiddle[15]
	w1 = complex(real(w1), -imag(w1))
	w2 = complex(real(w2), -imag(w2))
	w3 = complex(real(w3), -imag(w3))
	w4 = complex(real(w4), -imag(w4))
	w5 = complex(real(w5), -imag(w5))
	w6 = complex(real(w6), -imag(w6))
	w7 = complex(real(w7), -imag(w7))
	w8 = complex(real(w8), -imag(w8))
	w9 = complex(real(w9), -imag(w9))
	w10 = complex(real(w10), -imag(w10))
	w11 = complex(real(w11), -imag(w11))
	w12 = complex(real(w12), -imag(w12))
	w13 = complex(real(w13), -imag(w13))
	w14 = complex(real(w14), -imag(w14))
	w15 = complex(real(w15), -imag(w15))

	// Stage 1: 16 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Reorder input using bit-reversal indices during the first stage loads.
	x0 := s[br[0]]
	x1 := s[br[1]]
	a0, a1 := x0+x1, x0-x1
	x0 = s[br[2]]
	x1 = s[br[3]]
	a2, a3 := x0+x1, x0-x1
	x0 = s[br[4]]
	x1 = s[br[5]]
	a4, a5 := x0+x1, x0-x1
	x0 = s[br[6]]
	x1 = s[br[7]]
	a6, a7 := x0+x1, x0-x1
	x0 = s[br[8]]
	x1 = s[br[9]]
	a8, a9 := x0+x1, x0-x1
	x0 = s[br[10]]
	x1 = s[br[11]]
	a10, a11 := x0+x1, x0-x1
	x0 = s[br[12]]
	x1 = s[br[13]]
	a12, a13 := x0+x1, x0-x1
	x0 = s[br[14]]
	x1 = s[br[15]]
	a14, a15 := x0+x1, x0-x1
	x0 = s[br[16]]
	x1 = s[br[17]]
	a16, a17 := x0+x1, x0-x1
	x0 = s[br[18]]
	x1 = s[br[19]]
	a18, a19 := x0+x1, x0-x1
	x0 = s[br[20]]
	x1 = s[br[21]]
	a20, a21 := x0+x1, x0-x1
	x0 = s[br[22]]
	x1 = s[br[23]]
	a22, a23 := x0+x1, x0-x1
	x0 = s[br[24]]
	x1 = s[br[25]]
	a24, a25 := x0+x1, x0-x1
	x0 = s[br[26]]
	x1 = s[br[27]]
	a26, a27 := x0+x1, x0-x1
	x0 = s[br[28]]
	x1 = s[br[29]]
	a28, a29 := x0+x1, x0-x1
	x0 = s[br[30]]
	x1 = s[br[31]]
	a30, a31 := x0+x1, x0-x1

	// Stage 2: 8 radix-2 butterflies, stride=4
	b0, b2 := a0+a2, a0-a2
	t := w8 * a3
	b1, b3 := a1+t, a1-t
	b4, b6 := a4+a6, a4-a6
	t = w8 * a7
	b5, b7 := a5+t, a5-t
	b8, b10 := a8+a10, a8-a10
	t = w8 * a11
	b9, b11 := a9+t, a9-t
	b12, b14 := a12+a14, a12-a14
	t = w8 * a15
	b13, b15 := a13+t, a13-t
	b16, b18 := a16+a18, a16-a18
	t = w8 * a19
	b17, b19 := a17+t, a17-t
	b20, b22 := a20+a22, a20-a22
	t = w8 * a23
	b21, b23 := a21+t, a21-t
	b24, b26 := a24+a26, a24-a26
	t = w8 * a27
	b25, b27 := a25+t, a25-t
	b28, b30 := a28+a30, a28-a30
	t = w8 * a31
	b29, b31 := a29+t, a29-t

	// Stage 3: 4 radix-2 butterflies, stride=8
	c0, c4 := b0+b4, b0-b4
	t = w4 * b5
	c1, c5 := b1+t, b1-t
	t = w8 * b6
	c2, c6 := b2+t, b2-t
	t = w12 * b7
	c3, c7 := b3+t, b3-t
	c8, c12 := b8+b12, b8-b12
	t = w4 * b13
	c9, c13 := b9+t, b9-t
	t = w8 * b14
	c10, c14 := b10+t, b10-t
	t = w12 * b15
	c11, c15 := b11+t, b11-t
	c16, c20 := b16+b20, b16-b20
	t = w4 * b21
	c17, c21 := b17+t, b17-t
	t = w8 * b22
	c18, c22 := b18+t, b18-t
	t = w12 * b23
	c19, c23 := b19+t, b19-t
	c24, c28 := b24+b28, b24-b28
	t = w4 * b29
	c25, c29 := b25+t, b25-t
	t = w8 * b30
	c26, c30 := b26+t, b26-t
	t = w12 * b31
	c27, c31 := b27+t, b27-t

	// Stage 4: 2 radix-2 butterflies, stride=16
	d0, d8 := c0+c8, c0-c8
	t = w2 * c9
	d1, d9 := c1+t, c1-t
	t = w4 * c10
	d2, d10 := c2+t, c2-t
	t = w6 * c11
	d3, d11 := c3+t, c3-t
	t = w8 * c12
	d4, d12 := c4+t, c4-t
	t = w10 * c13
	d5, d13 := c5+t, c5-t
	t = w12 * c14
	d6, d14 := c6+t, c6-t
	t = w14 * c15
	d7, d15 := c7+t, c7-t
	d16, d24 := c16+c24, c16-c24
	t = w2 * c25
	d17, d25 := c17+t, c17-t
	t = w4 * c26
	d18, d26 := c18+t, c18-t
	t = w6 * c27
	d19, d27 := c19+t, c19-t
	t = w8 * c28
	d20, d28 := c20+t, c20-t
	t = w10 * c29
	d21, d29 := c21+t, c21-t
	t = w12 * c30
	d22, d30 := c22+t, c22-t
	t = w14 * c31
	d23, d31 := c23+t, c23-t

	// Stage 5: 1 radix-2 butterfly, stride=32 (full array)
	// Write directly to output or scratch buffer to avoid aliasing.
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]
	work[0], work[16] = d0+d16, d0-d16
	t = w1 * d17
	work[1], work[17] = d1+t, d1-t
	t = w2 * d18
	work[2], work[18] = d2+t, d2-t
	t = w3 * d19
	work[3], work[19] = d3+t, d3-t
	t = w4 * d20
	work[4], work[20] = d4+t, d4-t
	t = w5 * d21
	work[5], work[21] = d5+t, d5-t
	t = w6 * d22
	work[6], work[22] = d6+t, d6-t
	t = w7 * d23
	work[7], work[23] = d7+t, d7-t
	t = w8 * d24
	work[8], work[24] = d8+t, d8-t
	t = w9 * d25
	work[9], work[25] = d9+t, d9-t
	t = w10 * d26
	work[10], work[26] = d10+t, d10-t
	t = w11 * d27
	work[11], work[27] = d11+t, d11-t
	t = w12 * d28
	work[12], work[28] = d12+t, d12-t
	t = w13 * d29
	work[13], work[29] = d13+t, d13-t
	t = w14 * d30
	work[14], work[30] = d14+t, d14-t
	t = w15 * d31
	work[15], work[31] = d15+t, d15-t

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
