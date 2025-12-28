package fft

// forwardDIT256Complex64 computes a 256-point forward FFT using the
// Decimation-in-Time (DIT) Cooley-Tukey algorithm for complex64 data.
// The algorithm performs 8 stages of butterfly operations (log2(256) = 8).
// Optimized with partial loop unrolling and pre-loaded twiddle factors.
// Returns false if any slice is too small.
func forwardDIT256Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Use scratch buffer for in-place transforms to avoid aliasing issues
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hint for compiler optimization
	work = work[:n]
	src = src[:n]
	tw := twiddle[:n]
	br := bitrev[:n]

	// Pre-load frequently used twiddle factors
	w64 := tw[64] // Stage 2: j=1
	w32 := tw[32] // Stage 3: j=1
	w96 := tw[96] // Stage 3: j=3
	w16 := tw[16] // Stage 4
	w48 := tw[48]
	w80 := tw[80]
	w112 := tw[112]

	// Stage 1: 128 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1. Unroll by 4.
	for base := 0; base < n; base += 8 {
		a0, b0 := src[br[base]], src[br[base+1]]
		a1, b1 := src[br[base+2]], src[br[base+3]]
		a2, b2 := src[br[base+4]], src[br[base+5]]
		a3, b3 := src[br[base+6]], src[br[base+7]]
		work[base], work[base+1] = a0+b0, a0-b0
		work[base+2], work[base+3] = a1+b1, a1-b1
		work[base+4], work[base+5] = a2+b2, a2-b2
		work[base+6], work[base+7] = a3+b3, a3-b3
	}

	// Stage 2: 64 radix-2 butterflies, stride=4
	// Inner loop is only 2 iterations - unroll completely
	// j=0: twiddle[0]=1, j=1: twiddle[64]=w64
	for base := 0; base < n; base += 4 {
		// j=0: tw = 1
		x0, x2 := work[base], work[base+2]
		work[base], work[base+2] = x0+x2, x0-x2
		// j=1: tw = w64
		x1, x3 := work[base+1], work[base+3]
		t := w64 * x3
		work[base+1], work[base+3] = x1+t, x1-t
	}

	// Stage 3: 32 radix-2 butterflies, stride=8
	// Inner loop is 4 iterations - unroll completely
	// j=0: tw[0]=1, j=1: tw[32], j=2: tw[64], j=3: tw[96]
	for base := 0; base < n; base += 8 {
		// j=0: tw = 1
		x0, x4 := work[base], work[base+4]
		work[base], work[base+4] = x0+x4, x0-x4
		// j=1: tw = w32
		x1, x5 := work[base+1], work[base+5]
		t := w32 * x5
		work[base+1], work[base+5] = x1+t, x1-t
		// j=2: tw = w64
		x2, x6 := work[base+2], work[base+6]
		t = w64 * x6
		work[base+2], work[base+6] = x2+t, x2-t
		// j=3: tw = w96
		x3, x7 := work[base+3], work[base+7]
		t = w96 * x7
		work[base+3], work[base+7] = x3+t, x3-t
	}

	// Stage 4: 16 radix-2 butterflies, stride=16
	// Inner loop is 8 iterations - unroll completely
	for base := 0; base < n; base += 16 {
		// j=0: tw = 1
		x, y := work[base], work[base+8]
		work[base], work[base+8] = x+y, x-y
		// j=1: tw = w16
		x, y = work[base+1], work[base+9]
		t := w16 * y
		work[base+1], work[base+9] = x+t, x-t
		// j=2: tw = w32
		x, y = work[base+2], work[base+10]
		t = w32 * y
		work[base+2], work[base+10] = x+t, x-t
		// j=3: tw = w48
		x, y = work[base+3], work[base+11]
		t = w48 * y
		work[base+3], work[base+11] = x+t, x-t
		// j=4: tw = w64
		x, y = work[base+4], work[base+12]
		t = w64 * y
		work[base+4], work[base+12] = x+t, x-t
		// j=5: tw = w80
		x, y = work[base+5], work[base+13]
		t = w80 * y
		work[base+5], work[base+13] = x+t, x-t
		// j=6: tw = w96
		x, y = work[base+6], work[base+14]
		t = w96 * y
		work[base+6], work[base+14] = x+t, x-t
		// j=7: tw = w112
		x, y = work[base+7], work[base+15]
		t = w112 * y
		work[base+7], work[base+15] = x+t, x-t
	}

	// Stage 5: 8 radix-2 butterflies, stride=32
	// Unroll by 4 for better ILP
	for base := 0; base < n; base += 32 {
		for j := 0; j < 16; j += 4 {
			tw0, tw1 := tw[j*8], tw[(j+1)*8]
			tw2, tw3 := tw[(j+2)*8], tw[(j+3)*8]
			x0, y0 := work[base+j], work[base+j+16]
			x1, y1 := work[base+j+1], work[base+j+17]
			x2, y2 := work[base+j+2], work[base+j+18]
			x3, y3 := work[base+j+3], work[base+j+19]
			t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
			work[base+j], work[base+j+16] = x0+t0, x0-t0
			work[base+j+1], work[base+j+17] = x1+t1, x1-t1
			work[base+j+2], work[base+j+18] = x2+t2, x2-t2
			work[base+j+3], work[base+j+19] = x3+t3, x3-t3
		}
	}

	// Stage 6: 4 radix-2 butterflies, stride=64
	// Unroll by 4
	for base := 0; base < n; base += 64 {
		for j := 0; j < 32; j += 4 {
			tw0, tw1 := tw[j*4], tw[(j+1)*4]
			tw2, tw3 := tw[(j+2)*4], tw[(j+3)*4]
			x0, y0 := work[base+j], work[base+j+32]
			x1, y1 := work[base+j+1], work[base+j+33]
			x2, y2 := work[base+j+2], work[base+j+34]
			x3, y3 := work[base+j+3], work[base+j+35]
			t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
			work[base+j], work[base+j+32] = x0+t0, x0-t0
			work[base+j+1], work[base+j+33] = x1+t1, x1-t1
			work[base+j+2], work[base+j+34] = x2+t2, x2-t2
			work[base+j+3], work[base+j+35] = x3+t3, x3-t3
		}
	}

	// Stage 7: 2 radix-2 butterflies, stride=128
	// Unroll by 4
	for base := 0; base < n; base += 128 {
		for j := 0; j < 64; j += 4 {
			tw0, tw1 := tw[j*2], tw[(j+1)*2]
			tw2, tw3 := tw[(j+2)*2], tw[(j+3)*2]
			x0, y0 := work[base+j], work[base+j+64]
			x1, y1 := work[base+j+1], work[base+j+65]
			x2, y2 := work[base+j+2], work[base+j+66]
			x3, y3 := work[base+j+3], work[base+j+67]
			t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
			work[base+j], work[base+j+64] = x0+t0, x0-t0
			work[base+j+1], work[base+j+65] = x1+t1, x1-t1
			work[base+j+2], work[base+j+66] = x2+t2, x2-t2
			work[base+j+3], work[base+j+67] = x3+t3, x3-t3
		}
	}

	// Stage 8: 1 radix-2 butterfly, stride=256 (full array)
	// Unroll by 4
	for j := 0; j < 128; j += 4 {
		tw0, tw1, tw2, tw3 := tw[j], tw[j+1], tw[j+2], tw[j+3]
		x0, y0 := work[j], work[j+128]
		x1, y1 := work[j+1], work[j+129]
		x2, y2 := work[j+2], work[j+130]
		x3, y3 := work[j+3], work[j+131]
		t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
		work[j], work[j+128] = x0+t0, x0-t0
		work[j+1], work[j+129] = x1+t1, x1-t1
		work[j+2], work[j+130] = x2+t2, x2-t2
		work[j+3], work[j+131] = x3+t3, x3-t3
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	return true
}

// inverseDIT256Complex64 computes a 256-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Optimized with partial loop unrolling.
// Returns false if any slice is too small.
func inverseDIT256Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Use scratch buffer for in-place transforms to avoid aliasing issues
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hint for compiler optimization
	work = work[:n]
	src = src[:n]
	tw := twiddle[:n]
	br := bitrev[:n]

	// Pre-load and conjugate frequently used twiddle factors
	w64 := complex(real(tw[64]), -imag(tw[64]))
	w32 := complex(real(tw[32]), -imag(tw[32]))
	w96 := complex(real(tw[96]), -imag(tw[96]))
	w16 := complex(real(tw[16]), -imag(tw[16]))
	w48 := complex(real(tw[48]), -imag(tw[48]))
	w80 := complex(real(tw[80]), -imag(tw[80]))
	w112 := complex(real(tw[112]), -imag(tw[112]))

	// Stage 1: 128 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1. Unroll by 4.
	for base := 0; base < n; base += 8 {
		a0, b0 := src[br[base]], src[br[base+1]]
		a1, b1 := src[br[base+2]], src[br[base+3]]
		a2, b2 := src[br[base+4]], src[br[base+5]]
		a3, b3 := src[br[base+6]], src[br[base+7]]
		work[base], work[base+1] = a0+b0, a0-b0
		work[base+2], work[base+3] = a1+b1, a1-b1
		work[base+4], work[base+5] = a2+b2, a2-b2
		work[base+6], work[base+7] = a3+b3, a3-b3
	}

	// Stage 2: 64 radix-2 butterflies, stride=4
	// Inner loop is only 2 iterations - unroll completely
	for base := 0; base < n; base += 4 {
		// j=0: tw = 1
		x0, x2 := work[base], work[base+2]
		work[base], work[base+2] = x0+x2, x0-x2
		// j=1: tw = conj(w64)
		x1, x3 := work[base+1], work[base+3]
		t := w64 * x3
		work[base+1], work[base+3] = x1+t, x1-t
	}

	// Stage 3: 32 radix-2 butterflies, stride=8
	// Inner loop is 4 iterations - unroll completely
	for base := 0; base < n; base += 8 {
		// j=0: tw = 1
		x0, x4 := work[base], work[base+4]
		work[base], work[base+4] = x0+x4, x0-x4
		// j=1: tw = conj(w32)
		x1, x5 := work[base+1], work[base+5]
		t := w32 * x5
		work[base+1], work[base+5] = x1+t, x1-t
		// j=2: tw = conj(w64)
		x2, x6 := work[base+2], work[base+6]
		t = w64 * x6
		work[base+2], work[base+6] = x2+t, x2-t
		// j=3: tw = conj(w96)
		x3, x7 := work[base+3], work[base+7]
		t = w96 * x7
		work[base+3], work[base+7] = x3+t, x3-t
	}

	// Stage 4: 16 radix-2 butterflies, stride=16
	// Inner loop is 8 iterations - unroll completely
	for base := 0; base < n; base += 16 {
		// j=0: tw = 1
		x, y := work[base], work[base+8]
		work[base], work[base+8] = x+y, x-y
		// j=1: tw = conj(w16)
		x, y = work[base+1], work[base+9]
		t := w16 * y
		work[base+1], work[base+9] = x+t, x-t
		// j=2: tw = conj(w32)
		x, y = work[base+2], work[base+10]
		t = w32 * y
		work[base+2], work[base+10] = x+t, x-t
		// j=3: tw = conj(w48)
		x, y = work[base+3], work[base+11]
		t = w48 * y
		work[base+3], work[base+11] = x+t, x-t
		// j=4: tw = conj(w64)
		x, y = work[base+4], work[base+12]
		t = w64 * y
		work[base+4], work[base+12] = x+t, x-t
		// j=5: tw = conj(w80)
		x, y = work[base+5], work[base+13]
		t = w80 * y
		work[base+5], work[base+13] = x+t, x-t
		// j=6: tw = conj(w96)
		x, y = work[base+6], work[base+14]
		t = w96 * y
		work[base+6], work[base+14] = x+t, x-t
		// j=7: tw = conj(w112)
		x, y = work[base+7], work[base+15]
		t = w112 * y
		work[base+7], work[base+15] = x+t, x-t
	}

	// Stage 5: 8 radix-2 butterflies, stride=32
	// Unroll by 4 for better ILP
	for base := 0; base < n; base += 32 {
		for j := 0; j < 16; j += 4 {
			tw0 := complex(real(tw[j*8]), -imag(tw[j*8]))
			tw1 := complex(real(tw[(j+1)*8]), -imag(tw[(j+1)*8]))
			tw2 := complex(real(tw[(j+2)*8]), -imag(tw[(j+2)*8]))
			tw3 := complex(real(tw[(j+3)*8]), -imag(tw[(j+3)*8]))
			x0, y0 := work[base+j], work[base+j+16]
			x1, y1 := work[base+j+1], work[base+j+17]
			x2, y2 := work[base+j+2], work[base+j+18]
			x3, y3 := work[base+j+3], work[base+j+19]
			t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
			work[base+j], work[base+j+16] = x0+t0, x0-t0
			work[base+j+1], work[base+j+17] = x1+t1, x1-t1
			work[base+j+2], work[base+j+18] = x2+t2, x2-t2
			work[base+j+3], work[base+j+19] = x3+t3, x3-t3
		}
	}

	// Stage 6: 4 radix-2 butterflies, stride=64
	// Unroll by 4
	for base := 0; base < n; base += 64 {
		for j := 0; j < 32; j += 4 {
			tw0 := complex(real(tw[j*4]), -imag(tw[j*4]))
			tw1 := complex(real(tw[(j+1)*4]), -imag(tw[(j+1)*4]))
			tw2 := complex(real(tw[(j+2)*4]), -imag(tw[(j+2)*4]))
			tw3 := complex(real(tw[(j+3)*4]), -imag(tw[(j+3)*4]))
			x0, y0 := work[base+j], work[base+j+32]
			x1, y1 := work[base+j+1], work[base+j+33]
			x2, y2 := work[base+j+2], work[base+j+34]
			x3, y3 := work[base+j+3], work[base+j+35]
			t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
			work[base+j], work[base+j+32] = x0+t0, x0-t0
			work[base+j+1], work[base+j+33] = x1+t1, x1-t1
			work[base+j+2], work[base+j+34] = x2+t2, x2-t2
			work[base+j+3], work[base+j+35] = x3+t3, x3-t3
		}
	}

	// Stage 7: 2 radix-2 butterflies, stride=128
	// Unroll by 4
	for base := 0; base < n; base += 128 {
		for j := 0; j < 64; j += 4 {
			tw0 := complex(real(tw[j*2]), -imag(tw[j*2]))
			tw1 := complex(real(tw[(j+1)*2]), -imag(tw[(j+1)*2]))
			tw2 := complex(real(tw[(j+2)*2]), -imag(tw[(j+2)*2]))
			tw3 := complex(real(tw[(j+3)*2]), -imag(tw[(j+3)*2]))
			x0, y0 := work[base+j], work[base+j+64]
			x1, y1 := work[base+j+1], work[base+j+65]
			x2, y2 := work[base+j+2], work[base+j+66]
			x3, y3 := work[base+j+3], work[base+j+67]
			t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
			work[base+j], work[base+j+64] = x0+t0, x0-t0
			work[base+j+1], work[base+j+65] = x1+t1, x1-t1
			work[base+j+2], work[base+j+66] = x2+t2, x2-t2
			work[base+j+3], work[base+j+67] = x3+t3, x3-t3
		}
	}

	// Stage 8: 1 radix-2 butterfly, stride=256 (full array)
	// Unroll by 4
	for j := 0; j < 128; j += 4 {
		tw0 := complex(real(tw[j]), -imag(tw[j]))
		tw1 := complex(real(tw[j+1]), -imag(tw[j+1]))
		tw2 := complex(real(tw[j+2]), -imag(tw[j+2]))
		tw3 := complex(real(tw[j+3]), -imag(tw[j+3]))
		x0, y0 := work[j], work[j+128]
		x1, y1 := work[j+1], work[j+129]
		x2, y2 := work[j+2], work[j+130]
		x3, y3 := work[j+3], work[j+131]
		t0, t1, t2, t3 := tw0*y0, tw1*y1, tw2*y2, tw3*y3
		work[j], work[j+128] = x0+t0, x0-t0
		work[j+1], work[j+129] = x1+t1, x1-t1
		work[j+2], work[j+130] = x2+t2, x2-t2
		work[j+3], work[j+131] = x3+t3, x3-t3
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT256Complex128 computes a 256-point forward FFT using the
// Decimation-in-Time (DIT) Cooley-Tukey algorithm for complex128 data.
// The algorithm performs 8 stages of butterfly operations (log2(256) = 8).
// Returns false if any slice is too small.
func forwardDIT256Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Use scratch buffer for in-place transforms to avoid aliasing issues
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hint for compiler optimization
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Stage 1: 128 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1 to save one full pass over work.
	for base := 0; base < n; base += 2 {
		a := src[bitrev[base]]
		b := src[bitrev[base+1]]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stage 2: 64 radix-2 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*64]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	// Stage 3: 32 radix-2 butterflies, stride=8
	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	// Stage 4: 16 radix-2 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	// Stage 5: 8 radix-2 butterflies, stride=32
	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	// Stage 6: 4 radix-2 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	// Stage 7: 2 radix-2 butterflies, stride=128
	for base := 0; base < n; base += 128 {
		for j := range 64 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+64], tw)
			work[base+j] = a
			work[base+j+64] = b
		}
	}

	// Stage 8: 1 radix-2 butterfly, stride=256 (full array)
	for j := range 128 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+128], tw)
		work[j] = a
		work[j+128] = b
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	return true
}

// inverseDIT256Complex128 computes a 256-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Returns false if any slice is too small.
func inverseDIT256Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Use scratch buffer for in-place transforms to avoid aliasing issues
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hint for compiler optimization
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Stage 1: 128 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1 to save one full pass over work.
	for base := 0; base < n; base += 2 {
		a := src[bitrev[base]]
		b := src[bitrev[base+1]]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stage 2: 64 radix-2 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*64]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	// Stage 3: 32 radix-2 butterflies, stride=8
	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	// Stage 4: 16 radix-2 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	// Stage 5: 8 radix-2 butterflies, stride=32
	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	// Stage 6: 4 radix-2 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	// Stage 7: 2 radix-2 butterflies, stride=128
	for base := 0; base < n; base += 128 {
		for j := range 64 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+64], tw)
			work[base+j] = a
			work[base+j+64] = b
		}
	}

	// Stage 8: 1 radix-2 butterfly, stride=256 (full array)
	for j := range 128 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+128], tw)
		work[j] = a
		work[j+128] = b
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
