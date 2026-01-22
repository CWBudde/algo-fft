//go:build amd64 && asm && !purego

package kernels

import (
	m "github.com/MeKo-Christian/algo-fft/internal/math"
)

// Twiddle layout for AVX2 size-8192 radix-4-then-2 FFT:
//
// For size 8192 = 2 × 4^6, we have 7 stages:
//   Stage 1: 2048 radix-4 butterflies with bit-reversal (no packed twiddles needed, twiddle=1)
//   Stage 2: 512 groups × 4 butterflies, twiddle step = 512, j=0..3
//   Stage 3: 128 groups × 16 butterflies, twiddle step = 128, j=0..15
//   Stage 4: 32 groups × 64 butterflies, twiddle step = 32, j=0..63
//   Stage 5: 8 groups × 256 butterflies, twiddle step = 8, j=0..255
//   Stage 6: 2 groups × 1024 butterflies, twiddle step = 2, j=0..1023
//   Stage 7: radix-2, 4096 butterflies, twiddle indices 0..4095
//
// For each radix-4 butterfly in stages 2-6, we need w1, w2, w3 (3 twiddles).
// Each twiddle has real and imaginary parts, pre-broadcast for XMM (2 floats).
//
// Layout per radix-4 butterfly:
//   [w1.r, w1.r, w1.i, w1.i] - 16 bytes (XMM)
//   [w2.r, w2.r, w2.i, w2.i] - 16 bytes
//   [w3.r, w3.r, w3.i, w3.i] - 16 bytes
// Total: 48 bytes per radix-4 butterfly
//
// For radix-2 butterflies in stage 7, we need w (1 twiddle):
//   [w.r, w.r, w.i, w.i] - 16 bytes (XMM)
// Total: 16 bytes per radix-2 butterfly
//
// Twiddle size calculation:
//   Stage 2: 4 butterflies × 48 bytes = 192 bytes
//   Stage 3: 16 butterflies × 48 bytes = 768 bytes
//   Stage 4: 64 butterflies × 48 bytes = 3,072 bytes
//   Stage 5: 256 butterflies × 48 bytes = 12,288 bytes
//   Stage 6: 1024 butterflies × 48 bytes = 49,152 bytes
//   Stage 7: 4096 butterflies × 16 bytes = 65,536 bytes
//   Total: 131,008 bytes
//
// For inverse transform, we negate the imaginary parts (conjugate).

const (
	// twiddleSize8192Radix4Then2Bytes is the total twiddle size in bytes.
	twiddleSize8192Radix4Then2Bytes = 131008
	twiddleSize8192Radix4Then2Elems = twiddleSize8192Radix4Then2Bytes / 8

	// Stage offsets within the twiddle buffer (complex64 element offsets).
	twiddleStage2Offset8192 = 0
	twiddleStage3Offset8192 = 192 / 8
	twiddleStage4Offset8192 = 960 / 8
	twiddleStage5Offset8192 = 4032 / 8
	twiddleStage6Offset8192 = 16320 / 8
	twiddleStage7Offset8192 = 65472 / 8

	// elemsPerRadix4Butterfly is the element count for one radix-4 butterfly.
	elemsPerRadix4Butterfly = 6 // 3 twiddles × 2 complex64 each

	// elemsPerRadix2Butterfly is the element count for one radix-2 butterfly.
	elemsPerRadix2Butterfly = 2 // 1 twiddle × 2 complex64
)

// twiddleSize8192Radix4Then2AVX2 returns the element count for twiddles.
func twiddleSize8192Radix4Then2AVX2(_ int) int {
	return twiddleSize8192Radix4Then2Elems
}

// prepareTwiddle8192Radix4Then2AVX2 transforms standard twiddle factors into
// pre-broadcast SIMD twiddle data for the AVX2 size-8192 radix-4-then-2 kernel.
//
// For inverse transform, we use conjugates: W_8192^(-k) = conj(W_8192^k).
func prepareTwiddle8192Radix4Then2AVX2(n int, inverse bool, dst []complex64) {
	if n != 8192 || len(dst) < twiddleSize8192Radix4Then2Elems {
		return
	}

	twiddle := m.ComputeTwiddleFactors[complex64](n)

	// Stage 2: 4 butterflies (j=0..3), twiddle step = 512
	// Indices: j*512, 2*j*512, 3*j*512
	offset := twiddleStage2Offset8192
	for j := 0; j < 4; j++ {
		idx1 := j * 512
		idx2 := 2 * j * 512
		idx3 := 3 * j * 512
		writeTwiddle3Packed(dst[offset:], twiddle[idx1], twiddle[idx2], twiddle[idx3], inverse)
		offset += elemsPerRadix4Butterfly
	}

	// Stage 3: 16 butterflies (j=0..15), twiddle step = 128
	// Indices: j*128, 2*j*128, 3*j*128
	offset = twiddleStage3Offset8192
	for j := 0; j < 16; j++ {
		idx1 := j * 128
		idx2 := 2 * j * 128
		idx3 := 3 * j * 128
		writeTwiddle3Packed(dst[offset:], twiddle[idx1], twiddle[idx2], twiddle[idx3], inverse)
		offset += elemsPerRadix4Butterfly
	}

	// Stage 4: 64 butterflies (j=0..63), twiddle step = 32
	// Indices: j*32, 2*j*32, 3*j*32
	offset = twiddleStage4Offset8192
	for j := 0; j < 64; j++ {
		idx1 := j * 32
		idx2 := 2 * j * 32
		idx3 := 3 * j * 32
		writeTwiddle3Packed(dst[offset:], twiddle[idx1], twiddle[idx2], twiddle[idx3], inverse)
		offset += elemsPerRadix4Butterfly
	}

	// Stage 5: 256 butterflies (j=0..255), twiddle step = 8
	// Indices: j*8, 2*j*8, 3*j*8
	offset = twiddleStage5Offset8192
	for j := 0; j < 256; j++ {
		idx1 := j * 8
		idx2 := 2 * j * 8
		idx3 := 3 * j * 8
		writeTwiddle3Packed(dst[offset:], twiddle[idx1], twiddle[idx2], twiddle[idx3], inverse)
		offset += elemsPerRadix4Butterfly
	}

	// Stage 6: 1024 butterflies (j=0..1023), twiddle step = 2
	// Indices: j*2, 2*j*2, 3*j*2
	offset = twiddleStage6Offset8192
	for j := 0; j < 1024; j++ {
		idx1 := j * 2
		idx2 := 2 * j * 2
		idx3 := 3 * j * 2
		writeTwiddle3Packed(dst[offset:], twiddle[idx1], twiddle[idx2], twiddle[idx3], inverse)
		offset += elemsPerRadix4Butterfly
	}

	// Stage 7: 4096 radix-2 butterflies (j=0..4095)
	// Indices: j
	offset = twiddleStage7Offset8192
	for j := 0; j < 4096; j++ {
		writeTwiddle1Packed(dst[offset:], twiddle[j], inverse)
		offset += elemsPerRadix2Butterfly
	}
}

// writeTwiddle3Packed writes 3 twiddles (w1, w2, w3) to the packed buffer
// in SIMD-friendly format: [r, i, r, i] for each twiddle.
// This format works with VMOVSLDUP/VMOVSHDUP to broadcast r and i separately:
//
//	VMOVSLDUP [r, i, r, i] -> [r, r, r, r]
//	VMOVSHDUP [r, i, r, i] -> [i, i, i, i]
//
// If inverse is true, the imaginary parts are negated (conjugate).
func writeTwiddle3Packed(buf []complex64, w1, w2, w3 complex64, inverse bool) {
	sign := float32(1.0)
	if inverse {
		sign = -1.0
	}

	w1c := complex(real(w1), imag(w1)*sign)
	w2c := complex(real(w2), imag(w2)*sign)
	w3c := complex(real(w3), imag(w3)*sign)

	buf[0] = w1c
	buf[1] = w1c
	buf[2] = w2c
	buf[3] = w2c
	buf[4] = w3c
	buf[5] = w3c
}

// writeTwiddle1Packed writes 1 twiddle (w) to the packed buffer
// in SIMD-friendly format: [r, i, r, i].
// This format works with VMOVSLDUP/VMOVSHDUP to broadcast r and i separately:
//
//	VMOVSLDUP [r, i, r, i] -> [r, r, r, r]
//	VMOVSHDUP [r, i, r, i] -> [i, i, i, i]
//
// If inverse is true, the imaginary part is negated (conjugate).
func writeTwiddle1Packed(buf []complex64, w complex64, inverse bool) {
	sign := float32(1.0)
	if inverse {
		sign = -1.0
	}

	wc := complex(real(w), imag(w)*sign)
	buf[0] = wc
	buf[1] = wc
}

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
	// Each pair is 2 complex128 values = 32 bytes = 1 YMM register
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
		for pair := 0; pair < twiddlePairsPerCol256Radix16; pair++ {
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
	dst[offset+0] = w16(0)  // [W^0, W^1] for a1
	dst[offset+1] = w16(1)
	dst[offset+2] = w16(0)  // [W^0, W^2] for a2
	dst[offset+3] = w16(2)
	dst[offset+4] = w16(0)  // [W^0, W^3] for a3
	dst[offset+5] = w16(3)
	// Butterflies 2+3: twiddles for a1, a2, a3
	dst[offset+6] = w16(2)  // [W^2, W^3] for a1
	dst[offset+7] = w16(3)
	dst[offset+8] = w16(4)  // [W^4, W^6] for a2
	dst[offset+9] = w16(6)
	dst[offset+10] = w16(6) // [W^6, W^9] for a3
	dst[offset+11] = w16(9)
}
