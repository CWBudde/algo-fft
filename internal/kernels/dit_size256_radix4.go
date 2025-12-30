package kernels

// forwardDIT256Radix4Complex64 is the optimized radix-4 DIT FFT
// for size-256, incorporating all known optimizations:
//
// 1. Fused bit-reversal with Stage 1 (load src[bitrev[i]] directly)
// 2. Pointer comparison instead of sameSlice()
// 3. Fully inlined complex arithmetic (no function calls)
// 4. Pre-loaded twiddle factors for Stage 2
// 5. Minimized temporary variables.
func forwardDIT256Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 64 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	// Process directly into a temporary buffer to avoid aliasing
	var stage1 [256]complex64

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly with inlined complex operations
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// mulNegI(t3) = complex(imag(t3), -real(t3))
		stage1[base+1] = t1 + complex(imag(t3), -real(t3))
		// mulI(t3) = complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 2: 16 groups × 4 butterflies each
	// Use scratch for next stage output
	var stage2 [256]complex64

	for base := 0; base < n; base += 16 {
		// Pre-load all twiddle factors for this group
		w1_0 := tw[0]
		w2_0 := tw[0]
		w3_0 := tw[0]

		w1_1 := tw[16]
		w2_1 := tw[32]
		w3_1 := tw[48]

		w1_2 := tw[32]
		w2_2 := tw[64]
		w3_2 := tw[96]

		w1_3 := tw[48]
		w2_3 := tw[96]
		w3_3 := tw[144]

		// Unroll j=0..3
		// j=0
		a0 := stage1[base]
		a1 := w1_0 * stage1[base+4]
		a2 := w2_0 * stage1[base+8]
		a3 := w3_0 * stage1[base+12]
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		stage2[base] = t0 + t2
		stage2[base+8] = t0 - t2
		stage2[base+4] = t1 + complex(imag(t3), -real(t3))
		stage2[base+12] = t1 + complex(-imag(t3), real(t3))

		// j=1
		a0 = stage1[base+1]
		a1 = w1_1 * stage1[base+5]
		a2 = w2_1 * stage1[base+9]
		a3 = w3_1 * stage1[base+13]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+1] = t0 + t2
		stage2[base+9] = t0 - t2
		stage2[base+5] = t1 + complex(imag(t3), -real(t3))
		stage2[base+13] = t1 + complex(-imag(t3), real(t3))

		// j=2
		a0 = stage1[base+2]
		a1 = w1_2 * stage1[base+6]
		a2 = w2_2 * stage1[base+10]
		a3 = w3_2 * stage1[base+14]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+2] = t0 + t2
		stage2[base+10] = t0 - t2
		stage2[base+6] = t1 + complex(imag(t3), -real(t3))
		stage2[base+14] = t1 + complex(-imag(t3), real(t3))

		// j=3
		a0 = stage1[base+3]
		a1 = w1_3 * stage1[base+7]
		a2 = w2_3 * stage1[base+11]
		a3 = w3_3 * stage1[base+15]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+3] = t0 + t2
		stage2[base+11] = t0 - t2
		stage2[base+7] = t1 + complex(imag(t3), -real(t3))
		stage2[base+15] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 3: 4 groups × 16 butterflies each
	var stage3 [256]complex64

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*4]
			w2 := tw[2*j*4]
			w3 := tw[3*j*4]

			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			a0 := stage2[idx0]
			a1 := w1 * stage2[idx1]
			a2 := w2 * stage2[idx2]
			a3 := w3 * stage2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage3[idx0] = t0 + t2
			stage3[idx2] = t0 - t2
			stage3[idx1] = t1 + complex(imag(t3), -real(t3))
			stage3[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 4: 1 group × 64 butterflies
	// Write directly to output or scratch to avoid aliasing
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		a0 := stage3[idx0]
		a1 := w1 * stage3[idx1]
		a2 := w2 * stage3[idx2]
		a3 := w3 * stage3[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(imag(t3), -real(t3))
		work[idx3] = t1 + complex(-imag(t3), real(t3))
	}

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT256Radix4Complex64 is the optimized radix-4 DIT inverse FFT
// for size-256, using conjugated twiddle factors and 1/N scaling.
func inverseDIT256Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 64 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [256]complex64

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// For inverse: mulI instead of mulNegI
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 16 groups × 4 butterflies each with conjugated twiddles
	var stage2 [256]complex64

	for base := 0; base < n; base += 16 {
		// Pre-load and conjugate twiddle factors
		w1_0 := tw[0]
		w2_0 := tw[0]
		w3_0 := tw[0]

		w1_1 := complex(real(tw[16]), -imag(tw[16]))
		w2_1 := complex(real(tw[32]), -imag(tw[32]))
		w3_1 := complex(real(tw[48]), -imag(tw[48]))

		w1_2 := complex(real(tw[32]), -imag(tw[32]))
		w2_2 := complex(real(tw[64]), -imag(tw[64]))
		w3_2 := complex(real(tw[96]), -imag(tw[96]))

		w1_3 := complex(real(tw[48]), -imag(tw[48]))
		w2_3 := complex(real(tw[96]), -imag(tw[96]))
		w3_3 := complex(real(tw[144]), -imag(tw[144]))

		// Unroll j=0..3
		// j=0
		a0 := stage1[base]
		a1 := w1_0 * stage1[base+4]
		a2 := w2_0 * stage1[base+8]
		a3 := w3_0 * stage1[base+12]
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		stage2[base] = t0 + t2
		stage2[base+8] = t0 - t2
		stage2[base+4] = t1 + complex(-imag(t3), real(t3))
		stage2[base+12] = t1 + complex(imag(t3), -real(t3))

		// j=1
		a0 = stage1[base+1]
		a1 = w1_1 * stage1[base+5]
		a2 = w2_1 * stage1[base+9]
		a3 = w3_1 * stage1[base+13]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+1] = t0 + t2
		stage2[base+9] = t0 - t2
		stage2[base+5] = t1 + complex(-imag(t3), real(t3))
		stage2[base+13] = t1 + complex(imag(t3), -real(t3))

		// j=2
		a0 = stage1[base+2]
		a1 = w1_2 * stage1[base+6]
		a2 = w2_2 * stage1[base+10]
		a3 = w3_2 * stage1[base+14]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+2] = t0 + t2
		stage2[base+10] = t0 - t2
		stage2[base+6] = t1 + complex(-imag(t3), real(t3))
		stage2[base+14] = t1 + complex(imag(t3), -real(t3))

		// j=3
		a0 = stage1[base+3]
		a1 = w1_3 * stage1[base+7]
		a2 = w2_3 * stage1[base+11]
		a3 = w3_3 * stage1[base+15]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+3] = t0 + t2
		stage2[base+11] = t0 - t2
		stage2[base+7] = t1 + complex(-imag(t3), real(t3))
		stage2[base+15] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 3: 4 groups × 16 butterflies each with conjugated twiddles
	var stage3 [256]complex64

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			a0 := stage2[idx0]
			a1 := w1 * stage2[idx1]
			a2 := w2 * stage2[idx2]
			a3 := w3 * stage2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage3[idx0] = t0 + t2
			stage3[idx2] = t0 - t2
			stage3[idx1] = t1 + complex(-imag(t3), real(t3))
			stage3[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 4: 1 group × 64 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		a0 := stage3[idx0]
		a1 := w1 * stage3[idx1]
		a2 := w2 * stage3[idx2]
		a3 := w3 * stage3[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(-imag(t3), real(t3))
		work[idx3] = t1 + complex(imag(t3), -real(t3))
	}

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

// forwardDIT256Radix4Complex128 is the optimized radix-4 DIT FFT
// for size-256 complex128 data, incorporating all known optimizations.
func forwardDIT256Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 64 radix-4 butterflies with fused bit-reversal
	var stage1 [256]complex128

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(imag(t3), -real(t3))
		stage1[base+3] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 2: 16 groups × 4 butterflies each
	var stage2 [256]complex128

	for base := 0; base < n; base += 16 {
		// Pre-load twiddle factors
		w1_0 := tw[0]
		w2_0 := tw[0]
		w3_0 := tw[0]

		w1_1 := tw[16]
		w2_1 := tw[32]
		w3_1 := tw[48]

		w1_2 := tw[32]
		w2_2 := tw[64]
		w3_2 := tw[96]

		w1_3 := tw[48]
		w2_3 := tw[96]
		w3_3 := tw[144]

		// Unroll j=0..3
		// j=0
		a0 := stage1[base]
		a1 := w1_0 * stage1[base+4]
		a2 := w2_0 * stage1[base+8]
		a3 := w3_0 * stage1[base+12]
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		stage2[base] = t0 + t2
		stage2[base+8] = t0 - t2
		stage2[base+4] = t1 + complex(imag(t3), -real(t3))
		stage2[base+12] = t1 + complex(-imag(t3), real(t3))

		// j=1
		a0 = stage1[base+1]
		a1 = w1_1 * stage1[base+5]
		a2 = w2_1 * stage1[base+9]
		a3 = w3_1 * stage1[base+13]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+1] = t0 + t2
		stage2[base+9] = t0 - t2
		stage2[base+5] = t1 + complex(imag(t3), -real(t3))
		stage2[base+13] = t1 + complex(-imag(t3), real(t3))

		// j=2
		a0 = stage1[base+2]
		a1 = w1_2 * stage1[base+6]
		a2 = w2_2 * stage1[base+10]
		a3 = w3_2 * stage1[base+14]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+2] = t0 + t2
		stage2[base+10] = t0 - t2
		stage2[base+6] = t1 + complex(imag(t3), -real(t3))
		stage2[base+14] = t1 + complex(-imag(t3), real(t3))

		// j=3
		a0 = stage1[base+3]
		a1 = w1_3 * stage1[base+7]
		a2 = w2_3 * stage1[base+11]
		a3 = w3_3 * stage1[base+15]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+3] = t0 + t2
		stage2[base+11] = t0 - t2
		stage2[base+7] = t1 + complex(imag(t3), -real(t3))
		stage2[base+15] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 3: 4 groups × 16 butterflies each
	var stage3 [256]complex128

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*4]
			w2 := tw[2*j*4]
			w3 := tw[3*j*4]

			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			a0 := stage2[idx0]
			a1 := w1 * stage2[idx1]
			a2 := w2 * stage2[idx2]
			a3 := w3 * stage2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage3[idx0] = t0 + t2
			stage3[idx2] = t0 - t2
			stage3[idx1] = t1 + complex(imag(t3), -real(t3))
			stage3[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 4: 1 group × 64 butterflies
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		a0 := stage3[idx0]
		a1 := w1 * stage3[idx1]
		a2 := w2 * stage3[idx2]
		a3 := w3 * stage3[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(imag(t3), -real(t3))
		work[idx3] = t1 + complex(-imag(t3), real(t3))
	}

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT256Radix4Complex128 is the optimized radix-4 DIT inverse FFT
// for size-256 complex128 data, using conjugated twiddle factors and 1/N scaling.
func inverseDIT256Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 64 radix-4 butterflies with fused bit-reversal
	var stage1 [256]complex128

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// For inverse: mulI instead of mulNegI
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 16 groups × 4 butterflies each with conjugated twiddles
	var stage2 [256]complex128

	for base := 0; base < n; base += 16 {
		// Pre-load and conjugate twiddle factors
		w1_0 := tw[0]
		w2_0 := tw[0]
		w3_0 := tw[0]

		w1_1 := complex(real(tw[16]), -imag(tw[16]))
		w2_1 := complex(real(tw[32]), -imag(tw[32]))
		w3_1 := complex(real(tw[48]), -imag(tw[48]))

		w1_2 := complex(real(tw[32]), -imag(tw[32]))
		w2_2 := complex(real(tw[64]), -imag(tw[64]))
		w3_2 := complex(real(tw[96]), -imag(tw[96]))

		w1_3 := complex(real(tw[48]), -imag(tw[48]))
		w2_3 := complex(real(tw[96]), -imag(tw[96]))
		w3_3 := complex(real(tw[144]), -imag(tw[144]))

		// Unroll j=0..3
		// j=0
		a0 := stage1[base]
		a1 := w1_0 * stage1[base+4]
		a2 := w2_0 * stage1[base+8]
		a3 := w3_0 * stage1[base+12]
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3
		stage2[base] = t0 + t2
		stage2[base+8] = t0 - t2
		stage2[base+4] = t1 + complex(-imag(t3), real(t3))
		stage2[base+12] = t1 + complex(imag(t3), -real(t3))

		// j=1
		a0 = stage1[base+1]
		a1 = w1_1 * stage1[base+5]
		a2 = w2_1 * stage1[base+9]
		a3 = w3_1 * stage1[base+13]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+1] = t0 + t2
		stage2[base+9] = t0 - t2
		stage2[base+5] = t1 + complex(-imag(t3), real(t3))
		stage2[base+13] = t1 + complex(imag(t3), -real(t3))

		// j=2
		a0 = stage1[base+2]
		a1 = w1_2 * stage1[base+6]
		a2 = w2_2 * stage1[base+10]
		a3 = w3_2 * stage1[base+14]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+2] = t0 + t2
		stage2[base+10] = t0 - t2
		stage2[base+6] = t1 + complex(-imag(t3), real(t3))
		stage2[base+14] = t1 + complex(imag(t3), -real(t3))

		// j=3
		a0 = stage1[base+3]
		a1 = w1_3 * stage1[base+7]
		a2 = w2_3 * stage1[base+11]
		a3 = w3_3 * stage1[base+15]
		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3
		stage2[base+3] = t0 + t2
		stage2[base+11] = t0 - t2
		stage2[base+7] = t1 + complex(-imag(t3), real(t3))
		stage2[base+15] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 3: 4 groups × 16 butterflies each with conjugated twiddles
	var stage3 [256]complex128

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			a0 := stage2[idx0]
			a1 := w1 * stage2[idx1]
			a2 := w2 * stage2[idx2]
			a3 := w3 * stage2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage3[idx0] = t0 + t2
			stage3[idx2] = t0 - t2
			stage3[idx1] = t1 + complex(-imag(t3), real(t3))
			stage3[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 4: 1 group × 64 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		a0 := stage3[idx0]
		a1 := w1 * stage3[idx1]
		a2 := w2 * stage3[idx2]
		a3 := w3 * stage3[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(-imag(t3), real(t3))
		work[idx3] = t1 + complex(imag(t3), -real(t3))
	}

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

// bitReversalRadix4 performs base-4 bit-reversal for radix-4 FFT
// For a value with 'digits' base-4 digits, reverse the digit order.
func bitReversalRadix4(x, digits int) int {
	result := 0
	for range digits {
		result = (result << 2) | (x & 0x3) // Extract 2 bits and shift result
		x >>= 2
	}

	return result
}

// ComputeBitReversalIndicesRadix4 precomputes radix-4 bit-reversal indices for size n.
// n must be a power of 4.
func ComputeBitReversalIndicesRadix4(n int) []int {
	if n <= 0 || (n&(n-1)) != 0 {
		return nil // Not a power of 2
	}

	// Calculate number of base-4 digits
	digits := 0

	temp := n
	for temp > 1 {
		if temp&0x3 != 0 {
			return nil // Not a power of 4
		}

		digits++
		temp >>= 2
	}

	indices := make([]int, n)
	for i := range n {
		indices[i] = bitReversalRadix4(i, digits)
	}

	return indices
}
