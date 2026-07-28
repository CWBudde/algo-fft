//go:build amd64 && !purego

package kernels

import (
	m "github.com/cwbudde/algo-fft/internal/math"
)

const (
	twiddleSize1024Radix32x32AVX2Elems = 1128

	twiddleStage2Offset1024 = 1024
	twiddleStage3Offset1024 = 1028
	twiddleStage4Offset1024 = 1040
	twiddleStage5Offset1024 = 1068

	twiddleStageEntryElems = 4
)

const (
	twiddleSize256Radix16AVX2Elems = 748 // 736 + 12 for Stage 2 pre-packed
	twiddleSize256Radix16BaseElems = 256
	twiddlePairsPerCol256Radix16   = 8
	twiddleElemsPerPair256Radix16  = 4

	// Pre-packed YMM pairs for FFT-16 Stage 2 (eliminates VINSERTF128 at runtime)
	// Each pair is 2 complex128 values = 32 bytes = 1 YMM register.
	twiddleStage2PackedOffset256Radix16 = 736
	twiddleStage2PackedElems256Radix16  = 12 // 6 pairs × 2 elements
)

func twiddleSize1024Radix32x32AVX2(_ int) int {
	return twiddleSize1024Radix32x32AVX2Elems
}

func prepareTwiddle1024Radix32x32AVX2(n int, inverse bool, dst []complex128) {
	if n != 1024 || len(dst) < twiddleSize1024Radix32x32AVX2Elems {
		return
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	copy(dst[:n], twiddle)

	sign := 1.0
	if inverse {
		sign = -1.0
	}

	writeStageEntry := func(offset int, w complex128) {
		re := real(w)
		im := imag(w) * sign
		dst[offset+0] = complex(re, re)
		dst[offset+1] = complex(re, re)
		dst[offset+2] = complex(im, im)
		dst[offset+3] = complex(im, im)
	}

	offset := twiddleStage2Offset1024
	for j := 1; j < 2; j++ {
		writeStageEntry(offset, twiddle[j*256])
		offset += twiddleStageEntryElems
	}

	offset = twiddleStage3Offset1024
	for j := 1; j < 4; j++ {
		writeStageEntry(offset, twiddle[j*128])
		offset += twiddleStageEntryElems
	}

	offset = twiddleStage4Offset1024
	for j := 1; j < 8; j++ {
		writeStageEntry(offset, twiddle[j*64])
		offset += twiddleStageEntryElems
	}

	offset = twiddleStage5Offset1024
	for j := 1; j < 16; j++ {
		writeStageEntry(offset, twiddle[j*32])
		offset += twiddleStageEntryElems
	}
}

func twiddleSize256Radix16AVX2(_ int) int {
	return twiddleSize256Radix16AVX2Elems
}

// prepareTwiddle256Radix16AVX2 precomputes packed twiddles for the AVX2 size-256
// radix-16 complex128 kernel. The first 256 entries are the standard twiddles;
// the packed region starts at twiddleSize256Radix16BaseElems. Layout per (col,pair):
//
//	[re0,re0,re1,re1] then [im0,im0,im1,im1] (complex128 entries).
//
// For inverse, the imaginary parts are negated (conjugate).
func prepareTwiddle256Radix16AVX2(n int, inverse bool, dst []complex128) {
	if n != 256 || len(dst) < twiddleSize256Radix16AVX2Elems {
		return
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	copy(dst[:n], twiddle)
	sign := 1.0
	if inverse {
		sign = -1.0
	}

	writePair := func(offset int, w0, w1 complex128) {
		r0 := real(w0)
		i0 := imag(w0) * sign
		r1 := real(w1)
		i1 := imag(w1) * sign
		dst[offset+0] = complex(r0, r0)
		dst[offset+1] = complex(r1, r1)
		dst[offset+2] = complex(i0, i0)
		dst[offset+3] = complex(i1, i1)
	}

	offset := twiddleSize256Radix16BaseElems
	for col := 1; col < 16; col++ {
		for pair := range twiddlePairsPerCol256Radix16 {
			row := pair * 2
			idx0 := row * col
			idx1 := (row + 1) * col
			writePair(offset, twiddle[idx0], twiddle[idx1])
			offset += twiddleElemsPerPair256Radix16
		}
	}

	// Pre-packed YMM pairs for FFT-16 Stage 2 butterflies
	// W_16[k] = W_256[k*16], stored as contiguous complex128 pairs
	// Format: [W^a, W^b] = [re_a, im_a, re_b, im_b] (32 bytes, loadable as YMM)
	w16 := func(k int) complex128 {
		w := twiddle[k*16]
		if inverse {
			return complex(real(w), -imag(w))
		}
		return w
	}

	offset = twiddleStage2PackedOffset256Radix16
	// Butterflies 0+1: twiddles for a1, a2, a3
	dst[offset+0] = w16(0) // [W^0, W^1] for a1
	dst[offset+1] = w16(1)
	dst[offset+2] = w16(0) // [W^0, W^2] for a2
	dst[offset+3] = w16(2)
	dst[offset+4] = w16(0) // [W^0, W^3] for a3
	dst[offset+5] = w16(3)
	// Butterflies 2+3: twiddles for a1, a2, a3
	dst[offset+6] = w16(2) // [W^2, W^3] for a1
	dst[offset+7] = w16(3)
	dst[offset+8] = w16(4) // [W^4, W^6] for a2
	dst[offset+9] = w16(6)
	dst[offset+10] = w16(6) // [W^6, W^9] for a3
	dst[offset+11] = w16(9)
}
