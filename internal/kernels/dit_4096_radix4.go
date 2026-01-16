package kernels

// forwardDIT4096Radix4Complex64 computes a 4096-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// 4096 = 4^6, so this uses 6 radix-4 stages instead of 12 radix-2 stages.
//
// Optimizations:
// 1. Fused bit-reversal with Stage 1
// 2. Pointer comparison for aliasing detection
// 3. Stack-allocated stage buffers
// 4. Fully inlined complex arithmetic.
func forwardDIT4096Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 4096

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize4096Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 1024 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [4096]complex64

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

	// Stages 2-5: Process with stack buffers
	current := stage1[:]

	var (
		stage2 [4096]complex64
		stage3 [4096]complex64
		stage4 [4096]complex64
		stage5 [4096]complex64
	)

	stages := []struct {
		size int
		step int
		next *[4096]complex64
	}{
		{16, 256, &stage2}, // Stage 2: 256 groups × 4 butterflies
		{64, 64, &stage3},  // Stage 3: 64 groups × 16 butterflies
		{256, 16, &stage4}, // Stage 4: 16 groups × 64 butterflies
		{1024, 4, &stage5}, // Stage 5: 4 groups × 256 butterflies
	}

	for _, st := range stages {
		quarter := st.size / 4
		for base := 0; base < n; base += st.size {
			for j := range quarter {
				w1 := tw[j*st.step]
				w2 := tw[2*j*st.step]
				w3 := tw[3*j*st.step]

				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				a0 := current[idx0]
				a1 := w1 * current[idx1]
				a2 := w2 * current[idx2]
				a3 := w3 * current[idx3]

				t0 := a0 + a2
				t1 := a0 - a2
				t2 := a1 + a3
				t3 := a1 - a3

				st.next[idx0] = t0 + t2
				st.next[idx2] = t0 - t2
				st.next[idx1] = t1 + complex(imag(t3), -real(t3))
				st.next[idx3] = t1 + complex(-imag(t3), real(t3))
			}
		}

		current = st.next[:]
	}

	// Stage 6: 1 group × 1024 butterflies (final stage)
	// Write directly to output
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 1024 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 1024
		idx2 := j + 2048
		idx3 := j + 3072

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

// inverseDIT4096Radix4Complex64 computes a 4096-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
func inverseDIT4096Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 4096

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize4096Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 1024 radix-4 butterflies with fused bit-reversal
	var stage1 [4096]complex64

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

	// Stages 2-5: Process with conjugated twiddles
	current := stage1[:]

	var (
		stage2 [4096]complex64
		stage3 [4096]complex64
		stage4 [4096]complex64
		stage5 [4096]complex64
	)

	stages := []struct {
		size int
		step int
		next *[4096]complex64
	}{
		{16, 256, &stage2},
		{64, 64, &stage3},
		{256, 16, &stage4},
		{1024, 4, &stage5},
	}

	for _, st := range stages {
		quarter := st.size / 4
		for base := 0; base < n; base += st.size {
			for j := range quarter {
				w1 := complex(real(tw[j*st.step]), -imag(tw[j*st.step]))
				w2 := complex(real(tw[2*j*st.step]), -imag(tw[2*j*st.step]))
				w3 := complex(real(tw[3*j*st.step]), -imag(tw[3*j*st.step]))

				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				a0 := current[idx0]
				a1 := w1 * current[idx1]
				a2 := w2 * current[idx2]
				a3 := w3 * current[idx3]

				t0 := a0 + a2
				t1 := a0 - a2
				t2 := a1 + a3
				t3 := a1 - a3

				st.next[idx0] = t0 + t2
				st.next[idx2] = t0 - t2
				st.next[idx1] = t1 + complex(-imag(t3), real(t3))
				st.next[idx3] = t1 + complex(imag(t3), -real(t3))
			}
		}

		current = st.next[:]
	}

	// Stage 6: Final stage with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 1024 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 1024
		idx2 := j + 2048
		idx3 := j + 3072

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
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT4096Radix4Complex128 computes a 4096-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT4096Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 4096

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize4096Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 1024 radix-4 butterflies with fused bit-reversal
	var stage1 [4096]complex128

	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(imag(t3), -real(t3))
		stage1[base+3] = t1 + complex(-imag(t3), real(t3))
	}

	// Stages 2-6 using loop
	current := stage1[:]

	var (
		stage2   [4096]complex128
		stage3   [4096]complex128
		stage4   [4096]complex128
		stage5   [4096]complex128
		stageOut [4096]complex128
	)

	stages := []struct {
		size int
		step int
		next *[4096]complex128
	}{
		{16, 256, &stage2},
		{64, 64, &stage3},
		{256, 16, &stage4},
		{1024, 4, &stage5},
		{4096, 1, &stageOut},
	}

	for _, st := range stages {
		quarter := st.size / 4
		for base := 0; base < n; base += st.size {
			for j := range quarter {
				w1 := tw[j*st.step]
				w2 := tw[2*j*st.step]
				w3 := tw[3*j*st.step]

				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				a0 := current[idx0]
				a1 := w1 * current[idx1]
				a2 := w2 * current[idx2]
				a3 := w3 * current[idx3]

				t0 := a0 + a2
				t1 := a0 - a2
				t2 := a1 + a3
				t3 := a1 - a3

				st.next[idx0] = t0 + t2
				st.next[idx2] = t0 - t2
				st.next[idx1] = t1 + complex(imag(t3), -real(t3))
				st.next[idx3] = t1 + complex(-imag(t3), real(t3))
			}
		}

		current = st.next[:]
	}

	// Copy to output
	if &dst[0] == &src[0] {
		copy(scratch[:n], current)
		copy(dst, scratch[:n])
	} else {
		copy(dst, current)
	}

	return true
}

// inverseDIT4096Radix4Complex128 computes a 4096-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT4096Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 4096

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize4096Radix4
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 1024 radix-4 butterflies with fused bit-reversal
	var stage1 [4096]complex128

	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stages 2-6 with conjugated twiddles
	current := stage1[:]

	var (
		stage2   [4096]complex128
		stage3   [4096]complex128
		stage4   [4096]complex128
		stage5   [4096]complex128
		stageOut [4096]complex128
	)

	stages := []struct {
		size int
		step int
		next *[4096]complex128
	}{
		{16, 256, &stage2},
		{64, 64, &stage3},
		{256, 16, &stage4},
		{1024, 4, &stage5},
		{4096, 1, &stageOut},
	}

	for _, st := range stages {
		quarter := st.size / 4
		for base := 0; base < n; base += st.size {
			for j := range quarter {
				w1 := complex(real(tw[j*st.step]), -imag(tw[j*st.step]))
				w2 := complex(real(tw[2*j*st.step]), -imag(tw[2*j*st.step]))
				w3 := complex(real(tw[3*j*st.step]), -imag(tw[3*j*st.step]))

				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				a0 := current[idx0]
				a1 := w1 * current[idx1]
				a2 := w2 * current[idx2]
				a3 := w3 * current[idx3]

				t0 := a0 + a2
				t1 := a0 - a2
				t2 := a1 + a3
				t3 := a1 - a3

				st.next[idx0] = t0 + t2
				st.next[idx2] = t0 - t2
				st.next[idx1] = t1 + complex(-imag(t3), real(t3))
				st.next[idx3] = t1 + complex(imag(t3), -real(t3))
			}
		}

		current = st.next[:]
	}

	// Copy to output and apply scaling
	scale := complex(1.0/float64(n), 0)
	if &dst[0] == &src[0] {
		for i := range n {
			scratch[i] = current[i] * scale
		}

		copy(dst, scratch[:n])
	} else {
		for i := range n {
			dst[i] = current[i] * scale
		}
	}

	return true
}
