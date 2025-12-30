package kernels

// forwardDIT64Radix4Complex64 computes a 64-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// This uses 3 stages of radix-4 butterflies instead of 6 stages of radix-2.
// Returns false if any slice is too small.
func forwardDIT64Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 64

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 16 radix-4 butterflies with fused bit-reversal
	var stage1 [64]complex64

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

	// Stage 2: 4 groups × 4 butterflies each
	var stage2 [64]complex64

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			w1 := tw[j*4]
			w2 := tw[2*j*4]
			w3 := tw[3*j*4]

			a0 := stage1[idx0]
			a1 := w1 * stage1[idx1]
			a2 := w2 * stage1[idx2]
			a3 := w3 * stage1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage2[idx0] = t0 + t2
			stage2[idx2] = t0 - t2
			stage2[idx1] = t1 + complex(imag(t3), -real(t3))
			stage2[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 3: 1 group × 16 butterflies
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 16 {
		idx0 := j
		idx1 := j + 16
		idx2 := j + 32
		idx3 := j + 48

		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		a0 := stage2[idx0]
		a1 := w1 * stage2[idx1]
		a2 := w2 * stage2[idx2]
		a3 := w3 * stage2[idx3]

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

// inverseDIT64Radix4Complex64 computes a 64-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors and applies 1/N scaling at the end.
// Returns false if any slice is too small.
func inverseDIT64Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 64

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 16 radix-4 butterflies with fused bit-reversal
	var stage1 [64]complex64

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

	// Stage 2: 4 groups × 4 butterflies each with conjugated twiddles
	var stage2 [64]complex64

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			a0 := stage1[idx0]
			a1 := w1 * stage1[idx1]
			a2 := w2 * stage1[idx2]
			a3 := w3 * stage1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage2[idx0] = t0 + t2
			stage2[idx2] = t0 - t2
			stage2[idx1] = t1 + complex(-imag(t3), real(t3))
			stage2[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 3: 1 group × 16 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 16 {
		idx0 := j
		idx1 := j + 16
		idx2 := j + 32
		idx3 := j + 48

		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		a0 := stage2[idx0]
		a1 := w1 * stage2[idx1]
		a2 := w2 * stage2[idx2]
		a3 := w3 * stage2[idx3]

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

// forwardDIT64Radix4Complex128 computes a 64-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
// This uses 3 stages of radix-4 butterflies instead of 6 stages of radix-2.
// Returns false if any slice is too small.
func forwardDIT64Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 64

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 16 radix-4 butterflies with fused bit-reversal
	var stage1 [64]complex128

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

	// Stage 2: 4 groups × 4 butterflies each
	var stage2 [64]complex128

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			w1 := tw[j*4]
			w2 := tw[2*j*4]
			w3 := tw[3*j*4]

			a0 := stage1[idx0]
			a1 := w1 * stage1[idx1]
			a2 := w2 * stage1[idx2]
			a3 := w3 * stage1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage2[idx0] = t0 + t2
			stage2[idx2] = t0 - t2
			stage2[idx1] = t1 + complex(imag(t3), -real(t3))
			stage2[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 3: 1 group × 16 butterflies
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 16 {
		idx0 := j
		idx1 := j + 16
		idx2 := j + 32
		idx3 := j + 48

		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		a0 := stage2[idx0]
		a1 := w1 * stage2[idx1]
		a2 := w2 * stage2[idx2]
		a3 := w3 * stage2[idx3]

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

// inverseDIT64Radix4Complex128 computes a 64-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors and applies 1/N scaling at the end.
// Returns false if any slice is too small.
func inverseDIT64Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 64

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 16 radix-4 butterflies with fused bit-reversal
	var stage1 [64]complex128

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

	// Stage 2: 4 groups × 4 butterflies each with conjugated twiddles
	var stage2 [64]complex128

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			w1 := complex(real(tw[j*4]), -imag(tw[j*4]))
			w2 := complex(real(tw[2*j*4]), -imag(tw[2*j*4]))
			w3 := complex(real(tw[3*j*4]), -imag(tw[3*j*4]))

			a0 := stage1[idx0]
			a1 := w1 * stage1[idx1]
			a2 := w2 * stage1[idx2]
			a3 := w3 * stage1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage2[idx0] = t0 + t2
			stage2[idx2] = t0 - t2
			stage2[idx1] = t1 + complex(-imag(t3), real(t3))
			stage2[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 3: 1 group × 16 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 16 {
		idx0 := j
		idx1 := j + 16
		idx2 := j + 32
		idx3 := j + 48

		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		a0 := stage2[idx0]
		a1 := w1 * stage2[idx1]
		a2 := w2 * stage2[idx2]
		a3 := w3 * stage2[idx3]

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
