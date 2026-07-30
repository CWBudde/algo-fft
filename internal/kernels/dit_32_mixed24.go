package kernels

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

// bitrev32Mixed24 pre-computes mixed-radix bit-reversal indices for a 32-point FFT.
//
//nolint:gochecknoglobals
var bitrev32Mixed24 = mathpkg.ComputeBitReversalIndicesMixed24(32)

// forwardDIT32MixedRadix24Complex64 computes a size-32 FFT using a mixed
// radix-4/4/2 decomposition (two radix-4 stages, then one radix-2 stage).
func forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := dst[:n]
	workIsDst := true

	if &dst[0] == &src[0] {
		work = scratch[:n]
		workIsDst = false
	}

	br := bitrev32Mixed24

	s := src[:n]
	for i := range n {
		work[i] = s[br[i]]
	}

	// Stage 1+2 fused: radix-4, size=4 (no twiddles).
	for base := 0; base < n; base += 4 {
		a0 := work[base+0]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		y0 := t0 + t2
		y2 := t0 - t2
		y1 := t1 + mulNegI(t3)
		y3 := t1 + mulI(t3)

		work[base+0] = y0
		work[base+1] = y1
		work[base+2] = y2
		work[base+3] = y3
	}

	// Stage 3+4 fused: radix-4, size=16 (twiddle step = 2).
	for _, base := range []int{0, 16} {
		// j=0
		a0 := work[base+0]
		a1 := work[base+4]
		a2 := work[base+8]
		a3 := work[base+12]
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		y0 := t0 + t2
		y2 := t0 - t2
		y1 := t1 + mulNegI(t3)
		y3 := t1 + mulI(t3)
		work[base+0] = y0
		work[base+4] = y1
		work[base+8] = y2
		work[base+12] = y3

		// j=1
		a0 = work[base+1]
		a1 = mathpkg.MulComplex64(twiddle[2], work[base+5])
		a2 = mathpkg.MulComplex64(twiddle[4], work[base+9])
		a3 = mathpkg.MulComplex64(twiddle[6], work[base+13])
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		y0 = t0 + t2
		y2 = t0 - t2
		y1 = t1 + mulNegI(t3)
		y3 = t1 + mulI(t3)
		work[base+1] = y0
		work[base+5] = y1
		work[base+9] = y2
		work[base+13] = y3

		// j=2
		a0 = work[base+2]
		a1 = mathpkg.MulComplex64(twiddle[4], work[base+6])
		a2 = mathpkg.MulComplex64(twiddle[8], work[base+10])
		a3 = mathpkg.MulComplex64(twiddle[12], work[base+14])
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		y0 = t0 + t2
		y2 = t0 - t2
		y1 = t1 + mulNegI(t3)
		y3 = t1 + mulI(t3)
		work[base+2] = y0
		work[base+6] = y1
		work[base+10] = y2
		work[base+14] = y3

		// j=3
		a0 = work[base+3]
		a1 = mathpkg.MulComplex64(twiddle[6], work[base+7])
		a2 = mathpkg.MulComplex64(twiddle[12], work[base+11])
		a3 = mathpkg.MulComplex64(twiddle[18], work[base+15])
		y0, y1, y2, y3 = butterfly4Forward(a0, a1, a2, a3)
		work[base+3] = y0
		work[base+7] = y1
		work[base+11] = y2
		work[base+15] = y3
	}

	// Final stage: radix-2, size=32 (unrolled).
	a0 := work[0]
	t := mathpkg.MulComplex64(twiddle[0], work[16])
	work[0] = a0 + t
	work[16] = a0 - t

	a0 = work[1]
	t = mathpkg.MulComplex64(twiddle[1], work[17])
	work[1] = a0 + t
	work[17] = a0 - t

	a0 = work[2]
	t = mathpkg.MulComplex64(twiddle[2], work[18])
	work[2] = a0 + t
	work[18] = a0 - t

	a0 = work[3]
	t = mathpkg.MulComplex64(twiddle[3], work[19])
	work[3] = a0 + t
	work[19] = a0 - t

	a0 = work[4]
	t = mathpkg.MulComplex64(twiddle[4], work[20])
	work[4] = a0 + t
	work[20] = a0 - t

	a0 = work[5]
	t = mathpkg.MulComplex64(twiddle[5], work[21])
	work[5] = a0 + t
	work[21] = a0 - t

	a0 = work[6]
	t = mathpkg.MulComplex64(twiddle[6], work[22])
	work[6] = a0 + t
	work[22] = a0 - t

	a0 = work[7]
	t = mathpkg.MulComplex64(twiddle[7], work[23])
	work[7] = a0 + t
	work[23] = a0 - t

	a0 = work[8]
	t = mathpkg.MulComplex64(twiddle[8], work[24])
	work[8] = a0 + t
	work[24] = a0 - t

	a0 = work[9]
	t = mathpkg.MulComplex64(twiddle[9], work[25])
	work[9] = a0 + t
	work[25] = a0 - t

	a0 = work[10]
	t = mathpkg.MulComplex64(twiddle[10], work[26])
	work[10] = a0 + t
	work[26] = a0 - t

	a0 = work[11]
	t = mathpkg.MulComplex64(twiddle[11], work[27])
	work[11] = a0 + t
	work[27] = a0 - t

	a0 = work[12]
	t = mathpkg.MulComplex64(twiddle[12], work[28])
	work[12] = a0 + t
	work[28] = a0 - t

	a0 = work[13]
	t = mathpkg.MulComplex64(twiddle[13], work[29])
	work[13] = a0 + t
	work[29] = a0 - t

	a0 = work[14]
	t = mathpkg.MulComplex64(twiddle[14], work[30])
	work[14] = a0 + t
	work[30] = a0 - t

	a0 = work[15]
	t = mathpkg.MulComplex64(twiddle[15], work[31])
	work[15] = a0 + t
	work[31] = a0 - t

	if !workIsDst {
		copy(dst[:n], work)
	}

	return true
}

// inverseDIT32MixedRadix24Complex64 computes the inverse size-32 FFT using the
// same radix-4/4/2 decomposition and applies 1/N scaling.
func inverseDIT32MixedRadix24Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := dst[:n]
	workIsDst := true

	if &dst[0] == &src[0] {
		work = scratch[:n]
		workIsDst = false
	}

	br := bitrev32Mixed24

	s := src[:n]
	for i := range n {
		work[i] = s[br[i]]
	}

	// Stage 1+2 fused: radix-4, size=4 (no twiddles).
	for base := 0; base < n; base += 4 {
		a0 := work[base+0]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		y0 := t0 + t2
		y2 := t0 - t2
		y1 := t1 + mulI(t3)
		y3 := t1 + mulNegI(t3)

		work[base+0] = y0
		work[base+1] = y1
		work[base+2] = y2
		work[base+3] = y3
	}

	// Stage 3+4 fused: radix-4, size=16 (twiddle step = 2).
	for _, base := range []int{0, 16} {
		// j=0
		a0 := work[base+0]
		a1 := work[base+4]
		a2 := work[base+8]
		a3 := work[base+12]
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		y0 := t0 + t2
		y2 := t0 - t2
		y1 := t1 + mulI(t3)
		y3 := t1 + mulNegI(t3)
		work[base+0] = y0
		work[base+4] = y1
		work[base+8] = y2
		work[base+12] = y3

		// j=1
		a0 = work[base+1]
		a1 = mathpkg.MulComplex64(conj(twiddle[2]), work[base+5])
		a2 = mathpkg.MulComplex64(conj(twiddle[4]), work[base+9])
		a3 = mathpkg.MulComplex64(conj(twiddle[6]), work[base+13])
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		y0 = t0 + t2
		y2 = t0 - t2
		y1 = t1 + mulI(t3)
		y3 = t1 + mulNegI(t3)
		work[base+1] = y0
		work[base+5] = y1
		work[base+9] = y2
		work[base+13] = y3

		// j=2
		a0 = work[base+2]
		a1 = mathpkg.MulComplex64(conj(twiddle[4]), work[base+6])
		a2 = mathpkg.MulComplex64(conj(twiddle[8]), work[base+10])
		a3 = mathpkg.MulComplex64(conj(twiddle[12]), work[base+14])
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		y0 = t0 + t2
		y2 = t0 - t2
		y1 = t1 + mulI(t3)
		y3 = t1 + mulNegI(t3)
		work[base+2] = y0
		work[base+6] = y1
		work[base+10] = y2
		work[base+14] = y3

		// j=3
		a0 = work[base+3]
		a1 = mathpkg.MulComplex64(conj(twiddle[6]), work[base+7])
		a2 = mathpkg.MulComplex64(conj(twiddle[12]), work[base+11])
		a3 = mathpkg.MulComplex64(conj(twiddle[18]), work[base+15])
		y0, y1, y2, y3 = butterfly4Inverse(a0, a1, a2, a3)
		work[base+3] = y0
		work[base+7] = y1
		work[base+11] = y2
		work[base+15] = y3
	}

	// Final stage: radix-2, size=32 (unrolled).
	a0 := work[0]
	t := mathpkg.MulComplex64(conj(twiddle[0]), work[16])
	work[0] = a0 + t
	work[16] = a0 - t

	a0 = work[1]
	t = mathpkg.MulComplex64(conj(twiddle[1]), work[17])
	work[1] = a0 + t
	work[17] = a0 - t

	a0 = work[2]
	t = mathpkg.MulComplex64(conj(twiddle[2]), work[18])
	work[2] = a0 + t
	work[18] = a0 - t

	a0 = work[3]
	t = mathpkg.MulComplex64(conj(twiddle[3]), work[19])
	work[3] = a0 + t
	work[19] = a0 - t

	a0 = work[4]
	t = mathpkg.MulComplex64(conj(twiddle[4]), work[20])
	work[4] = a0 + t
	work[20] = a0 - t

	a0 = work[5]
	t = mathpkg.MulComplex64(conj(twiddle[5]), work[21])
	work[5] = a0 + t
	work[21] = a0 - t

	a0 = work[6]
	t = mathpkg.MulComplex64(conj(twiddle[6]), work[22])
	work[6] = a0 + t
	work[22] = a0 - t

	a0 = work[7]
	t = mathpkg.MulComplex64(conj(twiddle[7]), work[23])
	work[7] = a0 + t
	work[23] = a0 - t

	a0 = work[8]
	t = mathpkg.MulComplex64(conj(twiddle[8]), work[24])
	work[8] = a0 + t
	work[24] = a0 - t

	a0 = work[9]
	t = mathpkg.MulComplex64(conj(twiddle[9]), work[25])
	work[9] = a0 + t
	work[25] = a0 - t

	a0 = work[10]
	t = mathpkg.MulComplex64(conj(twiddle[10]), work[26])
	work[10] = a0 + t
	work[26] = a0 - t

	a0 = work[11]
	t = mathpkg.MulComplex64(conj(twiddle[11]), work[27])
	work[11] = a0 + t
	work[27] = a0 - t

	a0 = work[12]
	t = mathpkg.MulComplex64(conj(twiddle[12]), work[28])
	work[12] = a0 + t
	work[28] = a0 - t

	a0 = work[13]
	t = mathpkg.MulComplex64(conj(twiddle[13]), work[29])
	work[13] = a0 + t
	work[29] = a0 - t

	a0 = work[14]
	t = mathpkg.MulComplex64(conj(twiddle[14]), work[30])
	work[14] = a0 + t
	work[30] = a0 - t

	a0 = work[15]
	t = mathpkg.MulComplex64(conj(twiddle[15]), work[31])
	work[15] = a0 + t
	work[31] = a0 - t

	if !workIsDst {
		copy(dst[:n], work)
	}

	scale := float32(1.0 / float64(n))
	for i := range n {
		dst[i] = complex(real(dst[i])*scale, imag(dst[i])*scale)
	}

	return true
}
