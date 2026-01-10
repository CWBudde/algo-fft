package kernels

// forwardDIT1024Radix4Complex64 computes a 1024-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// 1024 = 4^5, so this uses 5 radix-4 stages instead of 10 radix-2 stages.
//
// Optimizations:
// 1. Fused bit-reversal with Stage 1
// 2. Pointer comparison for aliasing detection
// 3. Fully inlined complex arithmetic
// 4. Stack-allocated stage buffers
// 5. Pre-loaded twiddle factors.
func forwardDIT1024Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 1024

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize1024Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 256 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [1024]complex64

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

	// Stage 2: 64 groups × 4 butterflies each
	var stage2 [1024]complex64

	for base := 0; base < n; base += 16 {
		// Pre-load twiddle factors
		w1_0 := tw[0]
		w2_0 := tw[0]
		w3_0 := tw[0]

		w1_1 := tw[64]
		w2_1 := tw[128]
		w3_1 := tw[192]

		w1_2 := tw[128]
		w2_2 := tw[256]
		w3_2 := tw[384]

		w1_3 := tw[192]
		w2_3 := tw[384]
		w3_3 := tw[576]

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

	// Stage 3: 16 groups × 16 butterflies each
	var stage3 [1024]complex64

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*16]
			w2 := tw[2*j*16]
			w3 := tw[3*j*16]

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

	// Stage 4: 4 groups × 64 butterflies each
	var stage4 [1024]complex64

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := tw[j*4]
			w2 := tw[2*j*4]
			w3 := tw[3*j*4]

			idx0 := base + j
			idx1 := base + j + 64
			idx2 := base + j + 128
			idx3 := base + j + 192

			a0 := stage3[idx0]
			a1 := w1 * stage3[idx1]
			a2 := w2 * stage3[idx2]
			a3 := w3 * stage3[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage4[idx0] = t0 + t2
			stage4[idx2] = t0 - t2
			stage4[idx1] = t1 + complex(imag(t3), -real(t3))
			stage4[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 5: 1 group × 256 butterflies (final stage)
	// Write directly to output
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 256
		idx2 := j + 512
		idx3 := j + 768

		a0 := stage4[idx0]
		a1 := w1 * stage4[idx1]
		a2 := w2 * stage4[idx2]
		a3 := w3 * stage4[idx3]

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

// inverseDIT1024Radix4Complex64 computes a 1024-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
func inverseDIT1024Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 1024

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize1024Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 256 radix-4 butterflies with fused bit-reversal
	var stage1 [1024]complex64

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly (inverse)
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

	// Stage 2: 64 groups × 4 butterflies each with conjugated twiddles
	var stage2 [1024]complex64

	for base := 0; base < n; base += 16 {
		// Pre-load and conjugate twiddle factors
		w1_0 := tw[0]
		w2_0 := tw[0]
		w3_0 := tw[0]

		w1_1 := complex(real(tw[64]), -imag(tw[64]))
		w2_1 := complex(real(tw[128]), -imag(tw[128]))
		w3_1 := complex(real(tw[192]), -imag(tw[192]))

		w1_2 := complex(real(tw[128]), -imag(tw[128]))
		w2_2 := complex(real(tw[256]), -imag(tw[256]))
		w3_2 := complex(real(tw[384]), -imag(tw[384]))

		w1_3 := complex(real(tw[192]), -imag(tw[192]))
		w2_3 := complex(real(tw[384]), -imag(tw[384]))
		w3_3 := complex(real(tw[576]), -imag(tw[576]))

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

	// Stage 3: 16 groups × 16 butterflies each with conjugated twiddles
	var stage3 [1024]complex64

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*16]), -imag(tw[j*16]))
			w2 := complex(real(tw[2*j*16]), -imag(tw[2*j*16]))
			w3 := complex(real(tw[3*j*16]), -imag(tw[3*j*16]))

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

	// Stage 4: 4 groups × 64 butterflies each with conjugated twiddles
	var stage4 [1024]complex64

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			idx0 := base + j
			idx1 := base + j + 64
			idx2 := base + j + 128
			idx3 := base + j + 192

			a0 := stage3[idx0]
			a1 := w1 * stage3[idx1]
			a2 := w2 * stage3[idx2]
			a3 := w3 * stage3[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage4[idx0] = t0 + t2
			stage4[idx2] = t0 - t2
			stage4[idx1] = t1 + complex(-imag(t3), real(t3))
			stage4[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 5: 1 group × 256 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 256
		idx2 := j + 512
		idx3 := j + 768

		a0 := stage4[idx0]
		a1 := w1 * stage4[idx1]
		a2 := w2 * stage4[idx2]
		a3 := w3 * stage4[idx3]

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

// forwardDIT1024Radix4Complex128 computes a 1024-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT1024Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 1024

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize1024Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 256 radix-4 butterflies with fused bit-reversal
	var stage1 [1024]complex128

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

	// Stages 2-5 using loop (same pattern as complex64 but not unrolled)
	current := stage1[:]

	var (
		stage2 [1024]complex128
		stage3 [1024]complex128
		stage4 [1024]complex128
	)

	stageConfigs := []struct {
		size    int
		quarter int
		step    int
		next    []complex128
	}{
		{16, 4, 64, stage2[:]},
		{64, 16, 16, stage3[:]},
		{256, 64, 4, stage4[:]},
	}

	for _, cfg := range stageConfigs {
		for base := 0; base < n; base += cfg.size {
			for j := range cfg.quarter {
				w1 := tw[j*cfg.step]
				w2 := tw[2*j*cfg.step]
				w3 := tw[3*j*cfg.step]

				idx0 := base + j
				idx1 := idx0 + cfg.quarter
				idx2 := idx1 + cfg.quarter
				idx3 := idx2 + cfg.quarter

				a0 := current[idx0]
				a1 := w1 * current[idx1]
				a2 := w2 * current[idx2]
				a3 := w3 * current[idx3]

				t0 := a0 + a2
				t1 := a0 - a2
				t2 := a1 + a3
				t3 := a1 - a3

				cfg.next[idx0] = t0 + t2
				cfg.next[idx2] = t0 - t2
				cfg.next[idx1] = t1 + complex(imag(t3), -real(t3))
				cfg.next[idx3] = t1 + complex(-imag(t3), real(t3))
			}
		}

		current = cfg.next
	}

	// Stage 5: Final stage
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 256
		idx2 := j + 512
		idx3 := j + 768

		a0 := current[idx0]
		a1 := w1 * current[idx1]
		a2 := w2 * current[idx2]
		a3 := w3 * current[idx3]

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

// inverseDIT1024Radix4Complex128 computes a 1024-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT1024Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 1024

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize1024Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 256 radix-4 butterflies with fused bit-reversal
	var stage1 [1024]complex128

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly (inverse)
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stages 2-5 with conjugated twiddles
	current := stage1[:]

	var (
		stage2 [1024]complex128
		stage3 [1024]complex128
		stage4 [1024]complex128
	)

	stageConfigs := []struct {
		size    int
		quarter int
		step    int
		next    []complex128
	}{
		{16, 4, 64, stage2[:]},
		{64, 16, 16, stage3[:]},
		{256, 64, 4, stage4[:]},
	}

	for _, cfg := range stageConfigs {
		for base := 0; base < n; base += cfg.size {
			for j := range cfg.quarter {
				w1 := complex(real(tw[j*cfg.step]), -imag(tw[j*cfg.step]))
				w2 := complex(real(tw[2*j*cfg.step]), -imag(tw[2*j*cfg.step]))
				w3 := complex(real(tw[3*j*cfg.step]), -imag(tw[3*j*cfg.step]))

				idx0 := base + j
				idx1 := idx0 + cfg.quarter
				idx2 := idx1 + cfg.quarter
				idx3 := idx2 + cfg.quarter

				a0 := current[idx0]
				a1 := w1 * current[idx1]
				a2 := w2 * current[idx2]
				a3 := w3 * current[idx3]

				t0 := a0 + a2
				t1 := a0 - a2
				t2 := a1 + a3
				t3 := a1 - a3

				cfg.next[idx0] = t0 + t2
				cfg.next[idx2] = t0 - t2
				cfg.next[idx1] = t1 + complex(-imag(t3), real(t3))
				cfg.next[idx3] = t1 + complex(imag(t3), -real(t3))
			}
		}

		current = cfg.next
	}

	// Stage 5: Final stage with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 256
		idx2 := j + 512
		idx3 := j + 768

		a0 := current[idx0]
		a1 := w1 * current[idx1]
		a2 := w2 * current[idx2]
		a3 := w3 * current[idx3]

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
