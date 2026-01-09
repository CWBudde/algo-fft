package kernels

// forwardDIT16384Radix4Complex64 computes a 16384-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// 16384 = 4^7, so this uses 7 radix-4 stages instead of 14 radix-2 stages.
//
// Optimizations:
// 1. Fused bit-reversal with Stage 1
// 2. Pointer comparison for aliasing detection
// 3. Stack-allocated stage buffers
// 4. Fully inlined complex arithmetic.
func forwardDIT16384Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16384

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4096 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [16384]complex64

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

	// Stages 2-6: Process with stack buffers
	current := stage1[:]

	var (
		stage2 [16384]complex64
		stage3 [16384]complex64
		stage4 [16384]complex64
		stage5 [16384]complex64
		stage6 [16384]complex64
	)

	stages := []struct {
		size int
		step int
		next *[16384]complex64
	}{
		{16, 1024, &stage2}, // Stage 2: 1024 groups × 4 butterflies
		{64, 256, &stage3},  // Stage 3: 256 groups × 16 butterflies
		{256, 64, &stage4},  // Stage 4: 64 groups × 64 butterflies
		{1024, 16, &stage5}, // Stage 5: 16 groups × 256 butterflies
		{4096, 4, &stage6},  // Stage 6: 4 groups × 1024 butterflies
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

	// Stage 7: final stage (1 group × 4096 butterflies)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	quarter := 4096
	for j := range quarter {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
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

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(imag(t3), -real(t3))
		work[idx3] = t1 + complex(-imag(t3), real(t3))
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT16384Radix4Complex64 computes a 16384-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
func inverseDIT16384Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16384

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4096 radix-4 butterflies with fused bit-reversal
	var stage1 [16384]complex64

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

	// Stages 2-6: Process with conjugated twiddles
	current := stage1[:]

	var (
		stage2 [16384]complex64
		stage3 [16384]complex64
		stage4 [16384]complex64
		stage5 [16384]complex64
		stage6 [16384]complex64
	)

	stages := []struct {
		size int
		step int
		next *[16384]complex64
	}{
		{16, 1024, &stage2},
		{64, 256, &stage3},
		{256, 64, &stage4},
		{1024, 16, &stage5},
		{4096, 4, &stage6},
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

	// Stage 7: final stage with conjugated twiddles and normalization
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	scale := complex64(1.0 / n)

	quarter := 4096
	for j := range quarter {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
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

		work[idx0] = scale * (t0 + t2)
		work[idx2] = scale * (t0 - t2)
		work[idx1] = scale * (t1 + complex(-imag(t3), real(t3)))
		work[idx3] = scale * (t1 + complex(imag(t3), -real(t3)))
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// forwardDIT16384Radix4Complex128 computes a 16384-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT16384Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16384

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	bufA := dst

	bufB := scratch
	if &dst[0] == &src[0] {
		bufA = scratch
		bufB = dst
	}

	// Stage 1: 4096 radix-4 butterflies with fused bit-reversal
	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		bufA[base] = t0 + t2
		bufA[base+2] = t0 - t2
		bufA[base+1] = t1 + complex(imag(t3), -real(t3))
		bufA[base+3] = t1 + complex(-imag(t3), real(t3))
	}

	// Stages 2-6
	current := bufA
	next := bufB

	sizes := [...]int{16, 64, 256, 1024, 4096}
	steps := [...]int{1024, 256, 64, 16, 4}

	for stage := range sizes {
		size := sizes[stage]
		step := steps[stage]

		quarter := size / 4
		for base := 0; base < n; base += size {
			for j := range quarter {
				w1 := tw[j*step]
				w2 := tw[2*j*step]
				w3 := tw[3*j*step]

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

				next[idx0] = t0 + t2
				next[idx2] = t0 - t2
				next[idx1] = t1 + complex(imag(t3), -real(t3))
				next[idx3] = t1 + complex(-imag(t3), real(t3))
			}
		}

		current, next = next, current
	}

	// Stage 7: final stage
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	quarter := 4096
	for j := range quarter {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
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

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(imag(t3), -real(t3))
		work[idx3] = t1 + complex(-imag(t3), real(t3))
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT16384Radix4Complex128 computes a 16384-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT16384Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16384

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	bufA := dst

	bufB := scratch
	if &dst[0] == &src[0] {
		bufA = scratch
		bufB = dst
	}

	// Stage 1: 4096 radix-4 butterflies with fused bit-reversal
	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		bufA[base] = t0 + t2
		bufA[base+2] = t0 - t2
		bufA[base+1] = t1 + complex(-imag(t3), real(t3))
		bufA[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stages 2-6 with conjugated twiddles
	current := bufA
	next := bufB

	sizes := [...]int{16, 64, 256, 1024, 4096}
	steps := [...]int{1024, 256, 64, 16, 4}

	for stage := range sizes {
		size := sizes[stage]
		step := steps[stage]

		quarter := size / 4
		for base := 0; base < n; base += size {
			for j := range quarter {
				w1 := complex(real(tw[j*step]), -imag(tw[j*step]))
				w2 := complex(real(tw[2*j*step]), -imag(tw[2*j*step]))
				w3 := complex(real(tw[3*j*step]), -imag(tw[3*j*step]))

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

				next[idx0] = t0 + t2
				next[idx2] = t0 - t2
				next[idx1] = t1 + complex(-imag(t3), real(t3))
				next[idx3] = t1 + complex(imag(t3), -real(t3))
			}
		}

		current, next = next, current
	}

	// Stage 7: final stage with conjugated twiddles and normalization
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	scale := complex128(1.0 / n)

	quarter := 4096
	for j := range quarter {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
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

		work[idx0] = scale * (t0 + t2)
		work[idx2] = scale * (t0 - t2)
		work[idx1] = scale * (t1 + complex(-imag(t3), real(t3)))
		work[idx3] = scale * (t1 + complex(imag(t3), -real(t3)))
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}
