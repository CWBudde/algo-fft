package kernels

// forwardDIT8192Mixed24Complex64 computes a 8192-point forward FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex64 data.
//
// For n = 8192 = 2 × 4^6, this uses 7 stages instead of 13:
//   - Stages 1-6: radix-4 (2048, 512, 128, 32, 8, 2 groups)
//   - Stage 7: radix-2 (final combination of two 4096-point halves)
//
// Expected speedup: ~40% over pure radix-2.
func forwardDIT8192Mixed24Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 8192

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrevSize8192Mixed24[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 2048 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	stage1 := make([]complex64, n)

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

	// Stage 2: 512 radix-4 groups × 4 butterflies each
	stage2 := make([]complex64, n)

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*512]
			w2 := tw[2*j*512]
			w3 := tw[3*j*512]

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

	// Stage 3: 128 radix-4 groups × 16 butterflies each
	stage3 := make([]complex64, n)

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*128]
			w2 := tw[2*j*128]
			w3 := tw[3*j*128]

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

	// Stage 4: 32 radix-4 groups × 64 butterflies each
	stage4 := make([]complex64, n)

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := tw[j*32]
			w2 := tw[2*j*32]
			w3 := tw[3*j*32]

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

	// Stage 5: 8 radix-4 groups × 256 butterflies each
	stage5 := make([]complex64, n)

	for base := 0; base < n; base += 1024 {
		for j := range 256 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]

			idx0 := base + j
			idx1 := idx0 + 256
			idx2 := idx0 + 512
			idx3 := idx0 + 768

			a0 := stage4[idx0]
			a1 := w1 * stage4[idx1]
			a2 := w2 * stage4[idx2]
			a3 := w3 * stage4[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage5[idx0] = t0 + t2
			stage5[idx2] = t0 - t2
			stage5[idx1] = t1 + complex(imag(t3), -real(t3))
			stage5[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 6: 2 radix-4 groups × 1024 butterflies each
	stage6 := make([]complex64, n)

	for base := 0; base < n; base += 4096 {
		for j := range 1024 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]

			idx0 := base + j
			idx1 := idx0 + 1024
			idx2 := idx0 + 2048
			idx3 := idx0 + 3072

			a0 := stage5[idx0]
			a1 := w1 * stage5[idx1]
			a2 := w2 * stage5[idx2]
			a3 := w3 * stage5[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage6[idx0] = t0 + t2
			stage6[idx2] = t0 - t2
			stage6[idx1] = t1 + complex(imag(t3), -real(t3))
			stage6[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 7: radix-2 final stage (combines two 4096-point halves)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 4096 {
		tw := tw[j]
		a := stage6[j]
		b := tw * stage6[j+4096]
		work[j] = a + b
		work[j+4096] = a - b
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8192Mixed24Complex64 computes a 8192-point inverse FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex64 data.
//
// Uses conjugated twiddle factors and applies 1/N scaling.
func inverseDIT8192Mixed24Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 8192

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	br := bitrevSize8192Mixed24[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 2048 radix-4 butterflies with fused bit-reversal
	stage1 := make([]complex64, n)

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

	// Stage 2: 512 radix-4 groups with conjugated twiddles
	stage2 := make([]complex64, n)

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := complex(real(tw[j*512]), -imag(tw[j*512]))
			w2 := complex(real(tw[2*j*512]), -imag(tw[2*j*512]))
			w3 := complex(real(tw[3*j*512]), -imag(tw[3*j*512]))

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

	// Stage 3: 128 radix-4 groups with conjugated twiddles
	stage3 := make([]complex64, n)

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*128]), -imag(tw[j*128]))
			w2 := complex(real(tw[2*j*128]), -imag(tw[2*j*128]))
			w3 := complex(real(tw[3*j*128]), -imag(tw[3*j*128]))

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

	// Stage 4: 32 radix-4 groups with conjugated twiddles
	stage4 := make([]complex64, n)

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := complex(real(tw[j*32]), -imag(tw[j*32]))
			w2 := complex(real(tw[2*j*32]), -imag(tw[2*j*32]))
			w3 := complex(real(tw[3*j*32]), -imag(tw[3*j*32]))

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

	// Stage 5: 8 radix-4 groups with conjugated twiddles
	stage5 := make([]complex64, n)

	for base := 0; base < n; base += 1024 {
		for j := range 256 {
			w1 := complex(real(tw[j*8]), -imag(tw[j*8]))
			w2 := complex(real(tw[2*j*8]), -imag(tw[2*j*8]))
			w3 := complex(real(tw[3*j*8]), -imag(tw[3*j*8]))

			idx0 := base + j
			idx1 := idx0 + 256
			idx2 := idx0 + 512
			idx3 := idx0 + 768

			a0 := stage4[idx0]
			a1 := w1 * stage4[idx1]
			a2 := w2 * stage4[idx2]
			a3 := w3 * stage4[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage5[idx0] = t0 + t2
			stage5[idx2] = t0 - t2
			stage5[idx1] = t1 + complex(-imag(t3), real(t3))
			stage5[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 6: 2 radix-4 groups with conjugated twiddles
	stage6 := make([]complex64, n)

	for base := 0; base < n; base += 4096 {
		for j := range 1024 {
			w1 := complex(real(tw[j*2]), -imag(tw[j*2]))
			w2 := complex(real(tw[2*j*2]), -imag(tw[2*j*2]))
			w3 := complex(real(tw[3*j*2]), -imag(tw[3*j*2]))

			idx0 := base + j
			idx1 := idx0 + 1024
			idx2 := idx0 + 2048
			idx3 := idx0 + 3072

			a0 := stage5[idx0]
			a1 := w1 * stage5[idx1]
			a2 := w2 * stage5[idx2]
			a3 := w3 * stage5[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage6[idx0] = t0 + t2
			stage6[idx2] = t0 - t2
			stage6[idx1] = t1 + complex(-imag(t3), real(t3))
			stage6[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 7: radix-2 final stage with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 4096 {
		tw := complex(real(tw[j]), -imag(tw[j]))
		a := stage6[j]
		b := tw * stage6[j+4096]
		work[j] = a + b
		work[j+4096] = a - b
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

// forwardDIT8192Mixed24Complex128 computes a 8192-point forward FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT8192Mixed24Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 8192

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	br := bitrevSize8192Mixed24[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 2048 radix-4 butterflies with fused bit-reversal
	stage1 := make([]complex128, n)

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

	// Stage 2: 512 radix-4 groups
	stage2 := make([]complex128, n)

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*512]
			w2 := tw[2*j*512]
			w3 := tw[3*j*512]

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

	// Stage 3: 128 radix-4 groups
	stage3 := make([]complex128, n)

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*128]
			w2 := tw[2*j*128]
			w3 := tw[3*j*128]

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

	// Stage 4: 32 radix-4 groups
	stage4 := make([]complex128, n)

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := tw[j*32]
			w2 := tw[2*j*32]
			w3 := tw[3*j*32]

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

	// Stage 5: 8 radix-4 groups
	stage5 := make([]complex128, n)

	for base := 0; base < n; base += 1024 {
		for j := range 256 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]

			idx0 := base + j
			idx1 := idx0 + 256
			idx2 := idx0 + 512
			idx3 := idx0 + 768

			a0 := stage4[idx0]
			a1 := w1 * stage4[idx1]
			a2 := w2 * stage4[idx2]
			a3 := w3 * stage4[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage5[idx0] = t0 + t2
			stage5[idx2] = t0 - t2
			stage5[idx1] = t1 + complex(imag(t3), -real(t3))
			stage5[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 6: 2 radix-4 groups
	stage6 := make([]complex128, n)

	for base := 0; base < n; base += 4096 {
		for j := range 1024 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]

			idx0 := base + j
			idx1 := idx0 + 1024
			idx2 := idx0 + 2048
			idx3 := idx0 + 3072

			a0 := stage5[idx0]
			a1 := w1 * stage5[idx1]
			a2 := w2 * stage5[idx2]
			a3 := w3 * stage5[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage6[idx0] = t0 + t2
			stage6[idx2] = t0 - t2
			stage6[idx1] = t1 + complex(imag(t3), -real(t3))
			stage6[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 7: radix-2 final stage
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 4096 {
		tw := tw[j]
		a := stage6[j]
		b := tw * stage6[j+4096]
		work[j] = a + b
		work[j+4096] = a - b
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT8192Mixed24Complex128 computes a 8192-point inverse FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT8192Mixed24Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 8192

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	br := bitrevSize8192Mixed24[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 2048 radix-4 butterflies with fused bit-reversal
	stage1 := make([]complex128, n)

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

	// Stage 2: 512 radix-4 groups with conjugated twiddles
	stage2 := make([]complex128, n)

	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := complex(real(tw[j*512]), -imag(tw[j*512]))
			w2 := complex(real(tw[2*j*512]), -imag(tw[2*j*512]))
			w3 := complex(real(tw[3*j*512]), -imag(tw[3*j*512]))

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

	// Stage 3: 128 radix-4 groups with conjugated twiddles
	stage3 := make([]complex128, n)

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := complex(real(tw[j*128]), -imag(tw[j*128]))
			w2 := complex(real(tw[2*j*128]), -imag(tw[2*j*128]))
			w3 := complex(real(tw[3*j*128]), -imag(tw[3*j*128]))

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

	// Stage 4: 32 radix-4 groups with conjugated twiddles
	stage4 := make([]complex128, n)

	for base := 0; base < n; base += 256 {
		for j := range 64 {
			w1 := complex(real(tw[j*32]), -imag(tw[j*32]))
			w2 := complex(real(tw[2*j*32]), -imag(tw[2*j*32]))
			w3 := complex(real(tw[3*j*32]), -imag(tw[3*j*32]))

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

	// Stage 5: 8 radix-4 groups with conjugated twiddles
	stage5 := make([]complex128, n)

	for base := 0; base < n; base += 1024 {
		for j := range 256 {
			w1 := complex(real(tw[j*8]), -imag(tw[j*8]))
			w2 := complex(real(tw[2*j*8]), -imag(tw[2*j*8]))
			w3 := complex(real(tw[3*j*8]), -imag(tw[3*j*8]))

			idx0 := base + j
			idx1 := idx0 + 256
			idx2 := idx0 + 512
			idx3 := idx0 + 768

			a0 := stage4[idx0]
			a1 := w1 * stage4[idx1]
			a2 := w2 * stage4[idx2]
			a3 := w3 * stage4[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage5[idx0] = t0 + t2
			stage5[idx2] = t0 - t2
			stage5[idx1] = t1 + complex(-imag(t3), real(t3))
			stage5[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 6: 2 radix-4 groups with conjugated twiddles
	stage6 := make([]complex128, n)

	for base := 0; base < n; base += 4096 {
		for j := range 1024 {
			w1 := complex(real(tw[j*2]), -imag(tw[j*2]))
			w2 := complex(real(tw[2*j*2]), -imag(tw[2*j*2]))
			w3 := complex(real(tw[3*j*2]), -imag(tw[3*j*2]))

			idx0 := base + j
			idx1 := idx0 + 1024
			idx2 := idx0 + 2048
			idx3 := idx0 + 3072

			a0 := stage5[idx0]
			a1 := w1 * stage5[idx1]
			a2 := w2 * stage5[idx2]
			a3 := w3 * stage5[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			stage6[idx0] = t0 + t2
			stage6[idx2] = t0 - t2
			stage6[idx1] = t1 + complex(-imag(t3), real(t3))
			stage6[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	// Stage 7: radix-2 final stage with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 4096 {
		tw := complex(real(tw[j]), -imag(tw[j]))
		a := stage6[j]
		b := tw * stage6[j+4096]
		work[j] = a + b
		work[j+4096] = a - b
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
