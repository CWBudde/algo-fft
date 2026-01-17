package kernels

// forwardDIT512Radix4Then2Complex64 computes a 512-point forward FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex64 data.
//
// This uses 5 stages instead of 9:
//   - Stages 1-4: radix-4 (128, 32, 8, 2 groups)
//   - Stage 5: radix-2 (final combination of two 256-point halves)
//
// Expected speedup: ~40% over pure radix-2.
func forwardDIT512Radix4Then2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize512Radix4Then2
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 128 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [512]complex64

	for base := 0; base < n; base += 4 {
		// Load with mixed-radix bit-reversal
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

	// Stage 2: 32 radix-4 groups × 4 butterflies each
	var stage2 [512]complex64

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*32]
			w2 := tw[2*j*32]
			w3 := tw[3*j*32]

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

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

	// Stage 3: 8 radix-4 groups × 16 butterflies each
	var stage3 [512]complex64

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

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

	// Stage 4: 2 radix-4 groups × 64 butterflies each
	var stage4 [512]complex64

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]

			idx0 := base + j
			idx1 := idx0 + 64
			idx2 := idx0 + 128
			idx3 := idx0 + 192

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

	// Stage 5: radix-2 final stage (combines two 256-point halves)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		tw := tw[j]
		a := stage4[j]
		b := tw * stage4[j+256]
		work[j] = a + b
		work[j+256] = a - b
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT512Radix4Then2Complex64 computes a 512-point inverse FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex64 data.
//
// Uses conjugated twiddle factors and applies 1/N scaling.
func inverseDIT512Radix4Then2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	br := bitrevSize512Radix4Then2
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 128 radix-4 butterflies with fused bit-reversal
	var stage1 [512]complex64

	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		// Inverse butterfly: swap mulI and mulNegI
		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 32 radix-4 groups with conjugated twiddles
	var stage2 [512]complex64

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := complex(real(tw[j*32]), -imag(tw[j*32]))
			w2 := complex(real(tw[2*j*32]), -imag(tw[2*j*32]))
			w3 := complex(real(tw[3*j*32]), -imag(tw[3*j*32]))

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

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

	// Stage 3: 8 radix-4 groups with conjugated twiddles
	var stage3 [512]complex64

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*8]), -imag(tw[j*8]))
			w2 := complex(real(tw[2*j*8]), -imag(tw[2*j*8]))
			w3 := complex(real(tw[3*j*8]), -imag(tw[3*j*8]))

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

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

	// Stage 4: 2 radix-4 groups with conjugated twiddles
	var stage4 [512]complex64

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := complex(real(tw[j*2]), -imag(tw[j*2]))
			w2 := complex(real(tw[2*j*2]), -imag(tw[2*j*2]))
			w3 := complex(real(tw[3*j*2]), -imag(tw[3*j*2]))

			idx0 := base + j
			idx1 := idx0 + 64
			idx2 := idx0 + 128
			idx3 := idx0 + 192

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

	// Stage 5: radix-2 final stage with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		tw := complex(real(tw[j]), -imag(tw[j]))
		a := stage4[j]
		b := tw * stage4[j+256]
		work[j] = a + b
		work[j+256] = a - b
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT512Radix4Then2Complex128 computes a 512-point forward FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT512Radix4Then2Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	br := bitrevSize512Radix4Then2
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 128 radix-4 butterflies with fused bit-reversal
	var stage1 [512]complex128

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

	// Stage 2: 32 radix-4 groups
	var stage2 [512]complex128

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*32]
			w2 := tw[2*j*32]
			w3 := tw[3*j*32]

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

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

	// Stage 3: 8 radix-4 groups
	var stage3 [512]complex128

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

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

	// Stage 4: 2 radix-4 groups
	var stage4 [512]complex128

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]

			idx0 := base + j
			idx1 := idx0 + 64
			idx2 := idx0 + 128
			idx3 := idx0 + 192

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

	// Stage 5: radix-2 final stage
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		tw := tw[j]
		a := stage4[j]
		b := tw * stage4[j+256]
		work[j] = a + b
		work[j+256] = a - b
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT512Radix4Then2Complex128 computes a 512-point inverse FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT512Radix4Then2Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	br := bitrevSize512Radix4Then2
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 128 radix-4 butterflies with fused bit-reversal
	var stage1 [512]complex128

	for base := 0; base < n; base += 4 {
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		// Inverse butterfly: swap mulI and mulNegI
		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 32 radix-4 groups with conjugated twiddles
	var stage2 [512]complex128

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := complex(real(tw[j*32]), -imag(tw[j*32]))
			w2 := complex(real(tw[2*j*32]), -imag(tw[2*j*32]))
			w3 := complex(real(tw[3*j*32]), -imag(tw[3*j*32]))

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

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

	// Stage 3: 8 radix-4 groups with conjugated twiddles
	var stage3 [512]complex128

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*8]), -imag(tw[j*8]))
			w2 := complex(real(tw[2*j*8]), -imag(tw[2*j*8]))
			w3 := complex(real(tw[3*j*8]), -imag(tw[3*j*8]))

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

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

	// Stage 4: 2 radix-4 groups with conjugated twiddles
	var stage4 [512]complex128

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := complex(real(tw[j*2]), -imag(tw[j*2]))
			w2 := complex(real(tw[2*j*2]), -imag(tw[2*j*2]))
			w3 := complex(real(tw[3*j*2]), -imag(tw[3*j*2]))

			idx0 := base + j
			idx1 := idx0 + 64
			idx2 := idx0 + 128
			idx3 := idx0 + 192

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

	// Stage 5: radix-2 final stage with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 256 {
		tw := complex(real(tw[j]), -imag(tw[j]))
		a := stage4[j]
		b := tw * stage4[j+256]
		work[j] = a + b
		work[j+256] = a - b
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling
	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
