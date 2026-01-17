package kernels

// forwardDIT128Radix4Then2Complex64 computes a 128-point forward FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex64 data.
func forwardDIT128Radix4Then2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 32 radix-4 butterflies with fused bit-reversal.
	work1 := scratch[:n]
	work2 := dst[:n]

	for i := range 4 {
		a0 := s[0+i*8]
		a1 := s[32+i*8]
		a2 := s[64+i*8]
		a3 := s[96+i*8]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work1[0+i*4] = t0 + t2
		work1[2+i*4] = t0 - t2
		work1[1+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[3+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[4+i*8]
		a1 = s[36+i*8]
		a2 = s[68+i*8]
		a3 = s[100+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[32+i*4] = t0 + t2
		work1[34+i*4] = t0 - t2
		work1[33+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[35+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[2+i*8]
		a1 = s[34+i*8]
		a2 = s[66+i*8]
		a3 = s[98+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[16+i*4] = t0 + t2
		work1[18+i*4] = t0 - t2
		work1[17+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[19+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[6+i*8]
		a1 = s[38+i*8]
		a2 = s[70+i*8]
		a3 = s[102+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[48+i*4] = t0 + t2
		work1[50+i*4] = t0 - t2
		work1[49+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[51+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[1+i*8]
		a1 = s[33+i*8]
		a2 = s[65+i*8]
		a3 = s[97+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[64+i*4] = t0 + t2
		work1[66+i*4] = t0 - t2
		work1[65+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[67+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[5+i*8]
		a1 = s[37+i*8]
		a2 = s[69+i*8]
		a3 = s[101+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[96+i*4] = t0 + t2
		work1[98+i*4] = t0 - t2
		work1[97+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[99+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[3+i*8]
		a1 = s[35+i*8]
		a2 = s[67+i*8]
		a3 = s[99+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[80+i*4] = t0 + t2
		work1[82+i*4] = t0 - t2
		work1[81+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[83+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[7+i*8]
		a1 = s[39+i*8]
		a2 = s[71+i*8]
		a3 = s[103+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[112+i*4] = t0 + t2
		work1[114+i*4] = t0 - t2
		work1[113+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[115+i*4] = t1 + complex(-imag(t3), real(t3))
	}
	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

			a0 := work1[idx0]
			a1 := w1 * work1[idx1]
			a2 := w2 * work1[idx2]
			a3 := w3 * work1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work2[idx0] = t0 + t2
			work2[idx2] = t0 - t2
			work2[idx1] = t1 + complex(imag(t3), -real(t3))
			work2[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 3: 2 radix-4 groups × 16 butterflies each.
	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

			a0 := work2[idx0]
			a1 := w1 * work2[idx1]
			a2 := w2 * work2[idx2]
			a3 := w3 * work2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work1[idx0] = t0 + t2
			work1[idx2] = t0 - t2
			work1[idx1] = t1 + complex(imag(t3), -real(t3))
			work1[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	// Stage 4: radix-2 final stage (combines two 64-point halves).
	for j := range 64 {
		w := tw[j]
		a := work1[j]
		b := w * work1[j+64]
		work2[j] = a + b
		work2[j+64] = a - b
	}

	return true
}

// inverseDIT128Radix4Then2Complex64 computes a 128-point inverse FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex64 data.
func inverseDIT128Radix4Then2Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 32 radix-4 butterflies with fused bit-reversal.
	work1 := scratch[:n]
	work2 := dst[:n]

	for i := range 4 {
		a0 := s[0+i*8]
		a1 := s[32+i*8]
		a2 := s[64+i*8]
		a3 := s[96+i*8]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work1[0+i*4] = t0 + t2
		work1[2+i*4] = t0 - t2
		work1[1+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[3+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[4+i*8]
		a1 = s[36+i*8]
		a2 = s[68+i*8]
		a3 = s[100+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[32+i*4] = t0 + t2
		work1[34+i*4] = t0 - t2
		work1[33+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[35+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[2+i*8]
		a1 = s[34+i*8]
		a2 = s[66+i*8]
		a3 = s[98+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[16+i*4] = t0 + t2
		work1[18+i*4] = t0 - t2
		work1[17+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[19+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[6+i*8]
		a1 = s[38+i*8]
		a2 = s[70+i*8]
		a3 = s[102+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[48+i*4] = t0 + t2
		work1[50+i*4] = t0 - t2
		work1[49+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[51+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[1+i*8]
		a1 = s[33+i*8]
		a2 = s[65+i*8]
		a3 = s[97+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[64+i*4] = t0 + t2
		work1[66+i*4] = t0 - t2
		work1[65+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[67+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[5+i*8]
		a1 = s[37+i*8]
		a2 = s[69+i*8]
		a3 = s[101+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[96+i*4] = t0 + t2
		work1[98+i*4] = t0 - t2
		work1[97+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[99+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[3+i*8]
		a1 = s[35+i*8]
		a2 = s[67+i*8]
		a3 = s[99+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[80+i*4] = t0 + t2
		work1[82+i*4] = t0 - t2
		work1[81+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[83+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[7+i*8]
		a1 = s[39+i*8]
		a2 = s[71+i*8]
		a3 = s[103+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[112+i*4] = t0 + t2
		work1[114+i*4] = t0 - t2
		work1[113+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[115+i*4] = t1 + complex(imag(t3), -real(t3))
	}
	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]
			w1 = complex(real(w1), -imag(w1))
			w2 = complex(real(w2), -imag(w2))
			w3 = complex(real(w3), -imag(w3))

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

			a0 := work1[idx0]
			a1 := w1 * work1[idx1]
			a2 := w2 * work1[idx2]
			a3 := w3 * work1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work2[idx0] = t0 + t2
			work2[idx2] = t0 - t2
			work2[idx1] = t1 + complex(-imag(t3), real(t3))
			work2[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]
			w1 = complex(real(w1), -imag(w1))
			w2 = complex(real(w2), -imag(w2))
			w3 = complex(real(w3), -imag(w3))

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

			a0 := work2[idx0]
			a1 := w1 * work2[idx1]
			a2 := w2 * work2[idx2]
			a3 := w3 * work2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work1[idx0] = t0 + t2
			work1[idx2] = t0 - t2
			work1[idx1] = t1 + complex(-imag(t3), real(t3))
			work1[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	for j := range 64 {
		w := tw[j]
		w = complex(real(w), -imag(w))
		a := work1[j]
		b := w * work1[j+64]
		work2[j] = a + b
		work2[j+64] = a - b
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT128Radix4Then2Complex128 computes a 128-point forward FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT128Radix4Then2Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 32 radix-4 butterflies with fused bit-reversal.
	work1 := scratch[:n]
	work2 := dst[:n]

	for i := range 4 {
		a0 := s[0+i*8]
		a1 := s[32+i*8]
		a2 := s[64+i*8]
		a3 := s[96+i*8]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work1[0+i*4] = t0 + t2
		work1[2+i*4] = t0 - t2
		work1[1+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[3+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[4+i*8]
		a1 = s[36+i*8]
		a2 = s[68+i*8]
		a3 = s[100+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[32+i*4] = t0 + t2
		work1[34+i*4] = t0 - t2
		work1[33+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[35+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[2+i*8]
		a1 = s[34+i*8]
		a2 = s[66+i*8]
		a3 = s[98+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[16+i*4] = t0 + t2
		work1[18+i*4] = t0 - t2
		work1[17+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[19+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[6+i*8]
		a1 = s[38+i*8]
		a2 = s[70+i*8]
		a3 = s[102+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[48+i*4] = t0 + t2
		work1[50+i*4] = t0 - t2
		work1[49+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[51+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[1+i*8]
		a1 = s[33+i*8]
		a2 = s[65+i*8]
		a3 = s[97+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[64+i*4] = t0 + t2
		work1[66+i*4] = t0 - t2
		work1[65+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[67+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[5+i*8]
		a1 = s[37+i*8]
		a2 = s[69+i*8]
		a3 = s[101+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[96+i*4] = t0 + t2
		work1[98+i*4] = t0 - t2
		work1[97+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[99+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[3+i*8]
		a1 = s[35+i*8]
		a2 = s[67+i*8]
		a3 = s[99+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[80+i*4] = t0 + t2
		work1[82+i*4] = t0 - t2
		work1[81+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[83+i*4] = t1 + complex(-imag(t3), real(t3))

		a0 = s[7+i*8]
		a1 = s[39+i*8]
		a2 = s[71+i*8]
		a3 = s[103+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[112+i*4] = t0 + t2
		work1[114+i*4] = t0 - t2
		work1[113+i*4] = t1 + complex(imag(t3), -real(t3))
		work1[115+i*4] = t1 + complex(-imag(t3), real(t3))
	}
	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

			a0 := work1[idx0]
			a1 := w1 * work1[idx1]
			a2 := w2 * work1[idx2]
			a3 := w3 * work1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work2[idx0] = t0 + t2
			work2[idx2] = t0 - t2
			work2[idx1] = t1 + complex(imag(t3), -real(t3))
			work2[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

			a0 := work2[idx0]
			a1 := w1 * work2[idx1]
			a2 := w2 * work2[idx2]
			a3 := w3 * work2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work1[idx0] = t0 + t2
			work1[idx2] = t0 - t2
			work1[idx1] = t1 + complex(imag(t3), -real(t3))
			work1[idx3] = t1 + complex(-imag(t3), real(t3))
		}
	}

	for j := range 64 {
		w := tw[j]
		a := work1[j]
		b := w * work1[j+64]
		work2[j] = a + b
		work2[j+64] = a - b
	}

	return true
}

// inverseDIT128Radix4Then2Complex128 computes a 128-point inverse FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT128Radix4Then2Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 32 radix-4 butterflies with fused bit-reversal.
	work1 := scratch[:n]
	work2 := dst[:n]

	for i := range 4 {
		a0 := s[0+i*8]
		a1 := s[32+i*8]
		a2 := s[64+i*8]
		a3 := s[96+i*8]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work1[0+i*4] = t0 + t2
		work1[2+i*4] = t0 - t2
		work1[1+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[3+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[4+i*8]
		a1 = s[36+i*8]
		a2 = s[68+i*8]
		a3 = s[100+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[32+i*4] = t0 + t2
		work1[34+i*4] = t0 - t2
		work1[33+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[35+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[2+i*8]
		a1 = s[34+i*8]
		a2 = s[66+i*8]
		a3 = s[98+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[16+i*4] = t0 + t2
		work1[18+i*4] = t0 - t2
		work1[17+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[19+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[6+i*8]
		a1 = s[38+i*8]
		a2 = s[70+i*8]
		a3 = s[102+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[48+i*4] = t0 + t2
		work1[50+i*4] = t0 - t2
		work1[49+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[51+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[1+i*8]
		a1 = s[33+i*8]
		a2 = s[65+i*8]
		a3 = s[97+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[64+i*4] = t0 + t2
		work1[66+i*4] = t0 - t2
		work1[65+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[67+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[5+i*8]
		a1 = s[37+i*8]
		a2 = s[69+i*8]
		a3 = s[101+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[96+i*4] = t0 + t2
		work1[98+i*4] = t0 - t2
		work1[97+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[99+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[3+i*8]
		a1 = s[35+i*8]
		a2 = s[67+i*8]
		a3 = s[99+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[80+i*4] = t0 + t2
		work1[82+i*4] = t0 - t2
		work1[81+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[83+i*4] = t1 + complex(imag(t3), -real(t3))

		a0 = s[7+i*8]
		a1 = s[39+i*8]
		a2 = s[71+i*8]
		a3 = s[103+i*8]

		t0 = a0 + a2
		t1 = a0 - a2
		t2 = a1 + a3
		t3 = a1 - a3

		work1[112+i*4] = t0 + t2
		work1[114+i*4] = t0 - t2
		work1[113+i*4] = t1 + complex(-imag(t3), real(t3))
		work1[115+i*4] = t1 + complex(imag(t3), -real(t3))
	}
	for base := 0; base < n; base += 16 {
		for j := range 4 {
			w1 := tw[j*8]
			w2 := tw[2*j*8]
			w3 := tw[3*j*8]
			w1 = complex(real(w1), -imag(w1))
			w2 = complex(real(w2), -imag(w2))
			w3 = complex(real(w3), -imag(w3))

			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx0 + 8
			idx3 := idx0 + 12

			a0 := work1[idx0]
			a1 := w1 * work1[idx1]
			a2 := w2 * work1[idx2]
			a3 := w3 * work1[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work2[idx0] = t0 + t2
			work2[idx2] = t0 - t2
			work2[idx1] = t1 + complex(-imag(t3), real(t3))
			work2[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 16 {
			w1 := tw[j*2]
			w2 := tw[2*j*2]
			w3 := tw[3*j*2]
			w1 = complex(real(w1), -imag(w1))
			w2 = complex(real(w2), -imag(w2))
			w3 = complex(real(w3), -imag(w3))

			idx0 := base + j
			idx1 := idx0 + 16
			idx2 := idx0 + 32
			idx3 := idx0 + 48

			a0 := work2[idx0]
			a1 := w1 * work2[idx1]
			a2 := w2 * work2[idx2]
			a3 := w3 * work2[idx3]

			t0 := a0 + a2
			t1 := a0 - a2
			t2 := a1 + a3
			t3 := a1 - a3

			work1[idx0] = t0 + t2
			work1[idx2] = t0 - t2
			work1[idx1] = t1 + complex(-imag(t3), real(t3))
			work1[idx3] = t1 + complex(imag(t3), -real(t3))
		}
	}

	for j := range 64 {
		w := tw[j]
		w = complex(real(w), -imag(w))
		a := work1[j]
		b := w * work1[j+64]
		work2[j] = a + b
		work2[j+64] = a - b
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT128Radix4Then2Complex64 computes a 128-point forward FFT using
// radix-4-then-2 Decimation-in-Time (DIT) algorithm for complex64 data.
