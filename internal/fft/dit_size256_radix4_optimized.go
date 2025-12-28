package fft

// forwardDIT256Radix4OptimizedComplex64 is an optimized version of the radix-4
// DIT FFT for size-256, structured to encourage compiler vectorization.
//
// Key optimizations:
// - Minimized register pressure by reusing temporaries
// - Loop structures friendly to auto-vectorization
// - Explicit bounds elimination hints
// - Pre-extracted twiddle factors to enable better scheduling
func forwardDIT256Radix4OptimizedComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hints
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Bit-reversal with base-4 bit reversal
	for i := 0; i < n; i++ {
		work[i] = src[bitReversalRadix4(i, 4)]
	}

	// Stage 1: 64 butterflies, no twiddle multiplies (all W^0 = 1)
	// Fully unrolled for better performance
	for base := 0; base < n; base += 4 {
		a0 := work[base]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		// Inline butterfly4Forward for better optimization
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[base] = t0 + t2
		work[base+2] = t0 - t2
		work[base+1] = t1 + mulNegI(t3)
		work[base+3] = t1 + mulI(t3)
	}

	// Stage 2: 16 groups × 4 butterflies each
	// Twiddle step: 16
	for base := 0; base < n; base += 16 {
		// Pre-load twiddle factors for this group
		w1_0 := twiddle[0]
		w2_0 := twiddle[0]
		w3_0 := twiddle[0]

		w1_1 := twiddle[16]
		w2_1 := twiddle[32]
		w3_1 := twiddle[48]

		w1_2 := twiddle[32]
		w2_2 := twiddle[64]
		w3_2 := twiddle[96]

		w1_3 := twiddle[48]
		w2_3 := twiddle[96]
		w3_3 := twiddle[144]

		// j=0
		{
			a0 := work[base]
			a1 := w1_0 * work[base+4]
			a2 := w2_0 * work[base+8]
			a3 := w3_0 * work[base+12]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work[base] = t0 + t2
			work[base+8] = t0 - t2
			work[base+4] = t1 + mulNegI(t3)
			work[base+12] = t1 + mulI(t3)
		}

		// j=1
		{
			a0 := work[base+1]
			a1 := w1_1 * work[base+5]
			a2 := w2_1 * work[base+9]
			a3 := w3_1 * work[base+13]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work[base+1] = t0 + t2
			work[base+9] = t0 - t2
			work[base+5] = t1 + mulNegI(t3)
			work[base+13] = t1 + mulI(t3)
		}

		// j=2
		{
			a0 := work[base+2]
			a1 := w1_2 * work[base+6]
			a2 := w2_2 * work[base+10]
			a3 := w3_2 * work[base+14]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work[base+2] = t0 + t2
			work[base+10] = t0 - t2
			work[base+6] = t1 + mulNegI(t3)
			work[base+14] = t1 + mulI(t3)
		}

		// j=3
		{
			a0 := work[base+3]
			a1 := w1_3 * work[base+7]
			a2 := w2_3 * work[base+11]
			a3 := w3_3 * work[base+15]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work[base+3] = t0 + t2
			work[base+11] = t0 - t2
			work[base+7] = t1 + mulNegI(t3)
			work[base+15] = t1 + mulI(t3)
		}
	}

	// Stage 3: 4 groups × 16 butterflies each
	// Twiddle step: 4
	for base := 0; base < n; base += 64 {
		for j := 0; j < 16; j++ {
			w1 := twiddle[j*4]
			w2 := twiddle[2*j*4]
			w3 := twiddle[3*j*4]

			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			a0 := work[idx0]
			a1 := w1 * work[idx1]
			a2 := w2 * work[idx2]
			a3 := w3 * work[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work[idx0] = t0 + t2
			work[idx2] = t0 - t2
			work[idx1] = t1 + mulNegI(t3)
			work[idx3] = t1 + mulI(t3)
		}
	}

	// Stage 4: 1 group × 64 butterflies
	// Twiddle step: 1
	for j := 0; j < 64; j++ {
		w1 := twiddle[j]
		w2 := twiddle[2*j]
		w3 := twiddle[3*j]

		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		a0 := work[idx0]
		a1 := w1 * work[idx1]
		a2 := w2 * work[idx2]
		a3 := w3 * work[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + mulNegI(t3)
		work[idx3] = t1 + mulI(t3)
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}
