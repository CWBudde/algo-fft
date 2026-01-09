package kernels

// forwardDIT512Radix8Complex64 computes a 512-point forward FFT using
// pure radix-8 Decimation-in-Time (DIT) algorithm for complex64 data.
// 512 = 8 * 8 * 8 (3 stages).
func forwardDIT512Radix8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Internal twiddles for radix-8 butterfly
	w1_8 := tw[64]
	w2_8 := tw[128]
	w3_8 := tw[192]

	// Stage 1: 64 radix-8 butterflies with fused bit-reversal
	var stage1 [512]complex64

	for base := 0; base < n; base += 8 {
		x0 := s[br[base]]
		x1 := s[br[base+1]]
		x2 := s[br[base+2]]
		x3 := s[br[base+3]]
		x4 := s[br[base+4]]
		x5 := s[br[base+5]]
		x6 := s[br[base+6]]
		x7 := s[br[base+7]]

		// 8-point DFT (radix-8 butterfly)
		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(imag(a3), -real(a3))
		e3 := a1 + complex(-imag(a3), real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(imag(a7), -real(a7))
		o3 := a5 + complex(-imag(a7), real(a7))

		stage1[base] = e0 + o0
		stage1[base+4] = e0 - o0
		stage1[base+1] = e1 + w1_8*o1
		stage1[base+5] = e1 - w1_8*o1
		stage1[base+2] = e2 + w2_8*o2
		stage1[base+6] = e2 - w2_8*o2
		stage1[base+3] = e3 + w3_8*o3
		stage1[base+7] = e3 - w3_8*o3
	}

	// Stage 2: 8 groups, 8 radix-8 butterflies each
	var stage2 [512]complex64

	for base := 0; base < n; base += 64 {
		for j := range 8 {
			// Twiddles for this butterfly
			// W_512^{j * 8 * k} for k=1..7
			tw1 := tw[j*8]
			tw2 := tw[j*16]
			tw3 := tw[j*24]
			tw4 := tw[j*32]
			tw5 := tw[j*40]
			tw6 := tw[j*48]
			tw7 := tw[j*56]

			x0 := stage1[base+j]
			x1 := tw1 * stage1[base+j+8]
			x2 := tw2 * stage1[base+j+16]
			x3 := tw3 * stage1[base+j+24]
			x4 := tw4 * stage1[base+j+32]
			x5 := tw5 * stage1[base+j+40]
			x6 := tw6 * stage1[base+j+48]
			x7 := tw7 * stage1[base+j+56]

			// 8-point DFT
			a0 := x0 + x4
			a1 := x0 - x4
			a2 := x2 + x6
			a3 := x2 - x6
			a4 := x1 + x5
			a5 := x1 - x5
			a6 := x3 + x7
			a7 := x3 - x7

			e0 := a0 + a2
			e2 := a0 - a2
			e1 := a1 + complex(imag(a3), -real(a3))
			e3 := a1 + complex(-imag(a3), real(a3))

			o0 := a4 + a6
			o2 := a4 - a6
			o1 := a5 + complex(imag(a7), -real(a7))
			o3 := a5 + complex(-imag(a7), real(a7))

			stage2[base+j] = e0 + o0
			stage2[base+j+32] = e0 - o0
			stage2[base+j+8] = e1 + w1_8*o1
			stage2[base+j+40] = e1 - w1_8*o1
			stage2[base+j+16] = e2 + w2_8*o2
			stage2[base+j+48] = e2 - w2_8*o2
			stage2[base+j+24] = e3 + w3_8*o3
			stage2[base+j+56] = e3 - w3_8*o3
		}
	}

	// Stage 3: 1 group, 64 radix-8 butterflies each
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		// Twiddles for this butterfly
		// W_512^{j * 1 * k} for k=1..7
		tw1 := tw[j]
		tw2 := tw[2*j]
		tw3 := tw[3*j]
		tw4 := tw[4*j]
		tw5 := tw[5*j]
		tw6 := tw[6*j]
		tw7 := tw[7*j]

		x0 := stage2[j]
		x1 := tw1 * stage2[j+64]
		x2 := tw2 * stage2[j+128]
		x3 := tw3 * stage2[j+192]
		x4 := tw4 * stage2[j+256]
		x5 := tw5 * stage2[j+320]
		x6 := tw6 * stage2[j+384]
		x7 := tw7 * stage2[j+448]

		// 8-point DFT
		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(imag(a3), -real(a3))
		e3 := a1 + complex(-imag(a3), real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(imag(a7), -real(a7))
		o3 := a5 + complex(-imag(a7), real(a7))

		work[j] = e0 + o0
		work[j+256] = e0 - o0
		work[j+64] = e1 + w1_8*o1
		work[j+320] = e1 - w1_8*o1
		work[j+128] = e2 + w2_8*o2
		work[j+384] = e2 - w2_8*o2
		work[j+192] = e3 + w3_8*o3
		work[j+448] = e3 - w3_8*o3
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT512Radix8Complex64 computes a 512-point inverse FFT using
// pure radix-8 Decimation-in-Time (DIT) algorithm for complex64 data.
func inverseDIT512Radix8Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Internal twiddles for radix-8 butterfly (conjugated for inverse)
	w1_8 := complex(real(tw[64]), -imag(tw[64]))
	w2_8 := complex(real(tw[128]), -imag(tw[128]))
	w3_8 := complex(real(tw[192]), -imag(tw[192]))

	// Stage 1: 64 radix-8 butterflies with fused bit-reversal
	var stage1 [512]complex64

	for base := 0; base < n; base += 8 {
		x0 := s[br[base]]
		x1 := s[br[base+1]]
		x2 := s[br[base+2]]
		x3 := s[br[base+3]]
		x4 := s[br[base+4]]
		x5 := s[br[base+5]]
		x6 := s[br[base+6]]
		x7 := s[br[base+7]]

		// 8-point DFT (radix-8 butterfly) - inverse
		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(-imag(a3), real(a3))
		e3 := a1 + complex(imag(a3), -real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(-imag(a7), real(a7))
		o3 := a5 + complex(imag(a7), -real(a7))

		stage1[base] = e0 + o0
		stage1[base+4] = e0 - o0
		stage1[base+1] = e1 + w1_8*o1
		stage1[base+5] = e1 - w1_8*o1
		stage1[base+2] = e2 + w2_8*o2
		stage1[base+6] = e2 - w2_8*o2
		stage1[base+3] = e3 + w3_8*o3
		stage1[base+7] = e3 - w3_8*o3
	}

	// Stage 2: 8 groups, 8 radix-8 butterflies each
	var stage2 [512]complex64

	for base := 0; base < n; base += 64 {
		for j := range 8 {
			tw1 := complex(real(tw[j*8]), -imag(tw[j*8]))
			tw2 := complex(real(tw[j*16]), -imag(tw[j*16]))
			tw3 := complex(real(tw[j*24]), -imag(tw[j*24]))
			tw4 := complex(real(tw[j*32]), -imag(tw[j*32]))
			tw5 := complex(real(tw[j*40]), -imag(tw[j*40]))
			tw6 := complex(real(tw[j*48]), -imag(tw[j*48]))
			tw7 := complex(real(tw[j*56]), -imag(tw[j*56]))

			x0 := stage1[base+j]
			x1 := tw1 * stage1[base+j+8]
			x2 := tw2 * stage1[base+j+16]
			x3 := tw3 * stage1[base+j+24]
			x4 := tw4 * stage1[base+j+32]
			x5 := tw5 * stage1[base+j+40]
			x6 := tw6 * stage1[base+j+48]
			x7 := tw7 * stage1[base+j+56]

			a0 := x0 + x4
			a1 := x0 - x4
			a2 := x2 + x6
			a3 := x2 - x6
			a4 := x1 + x5
			a5 := x1 - x5
			a6 := x3 + x7
			a7 := x3 - x7

			e0 := a0 + a2
			e2 := a0 - a2
			e1 := a1 + complex(-imag(a3), real(a3))
			e3 := a1 + complex(imag(a3), -real(a3))

			o0 := a4 + a6
			o2 := a4 - a6
			o1 := a5 + complex(-imag(a7), real(a7))
			o3 := a5 + complex(imag(a7), -real(a7))

			stage2[base+j] = e0 + o0
			stage2[base+j+32] = e0 - o0
			stage2[base+j+8] = e1 + w1_8*o1
			stage2[base+j+40] = e1 - w1_8*o1
			stage2[base+j+16] = e2 + w2_8*o2
			stage2[base+j+48] = e2 - w2_8*o2
			stage2[base+j+24] = e3 + w3_8*o3
			stage2[base+j+56] = e3 - w3_8*o3
		}
	}

	// Stage 3: 1 group, 64 radix-8 butterflies each
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		tw1 := complex(real(tw[j]), -imag(tw[j]))
		tw2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		tw3 := complex(real(tw[3*j]), -imag(tw[3*j]))
		tw4 := complex(real(tw[4*j]), -imag(tw[4*j]))
		tw5 := complex(real(tw[5*j]), -imag(tw[5*j]))
		tw6 := complex(real(tw[6*j]), -imag(tw[6*j]))
		tw7 := complex(real(tw[7*j]), -imag(tw[7*j]))

		x0 := stage2[j]
		x1 := tw1 * stage2[j+64]
		x2 := tw2 * stage2[j+128]
		x3 := tw3 * stage2[j+192]
		x4 := tw4 * stage2[j+256]
		x5 := tw5 * stage2[j+320]
		x6 := tw6 * stage2[j+384]
		x7 := tw7 * stage2[j+448]

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(-imag(a3), real(a3))
		e3 := a1 + complex(imag(a3), -real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(-imag(a7), real(a7))
		o3 := a5 + complex(imag(a7), -real(a7))

		work[j] = e0 + o0
		work[j+256] = e0 - o0
		work[j+64] = e1 + w1_8*o1
		work[j+320] = e1 - w1_8*o1
		work[j+128] = e2 + w2_8*o2
		work[j+384] = e2 - w2_8*o2
		work[j+192] = e3 + w3_8*o3
		work[j+448] = e3 - w3_8*o3
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

// Complex128 versions

// forwardDIT512Radix8Complex128 computes a 512-point forward FFT using
// pure radix-8 Decimation-in-Time (DIT) algorithm for complex128 data.
func forwardDIT512Radix8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	w1_8 := tw[64]
	w2_8 := tw[128]
	w3_8 := tw[192]

	var stage1 [512]complex128

	for base := 0; base < n; base += 8 {
		x0 := s[br[base]]
		x1 := s[br[base+1]]
		x2 := s[br[base+2]]
		x3 := s[br[base+3]]
		x4 := s[br[base+4]]
		x5 := s[br[base+5]]
		x6 := s[br[base+6]]
		x7 := s[br[base+7]]

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(imag(a3), -real(a3))
		e3 := a1 + complex(-imag(a3), real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(imag(a7), -real(a7))
		o3 := a5 + complex(-imag(a7), real(a7))

		stage1[base] = e0 + o0
		stage1[base+4] = e0 - o0
		stage1[base+1] = e1 + w1_8*o1
		stage1[base+5] = e1 - w1_8*o1
		stage1[base+2] = e2 + w2_8*o2
		stage1[base+6] = e2 - w2_8*o2
		stage1[base+3] = e3 + w3_8*o3
		stage1[base+7] = e3 - w3_8*o3
	}

	var stage2 [512]complex128

	for base := 0; base < n; base += 64 {
		for j := range 8 {
			tw1 := tw[j*8]
			tw2 := tw[j*16]
			tw3 := tw[j*24]
			tw4 := tw[j*32]
			tw5 := tw[j*40]
			tw6 := tw[j*48]
			tw7 := tw[j*56]

			x0 := stage1[base+j]
			x1 := tw1 * stage1[base+j+8]
			x2 := tw2 * stage1[base+j+16]
			x3 := tw3 * stage1[base+j+24]
			x4 := tw4 * stage1[base+j+32]
			x5 := tw5 * stage1[base+j+40]
			x6 := tw6 * stage1[base+j+48]
			x7 := tw7 * stage1[base+j+56]

			a0 := x0 + x4
			a1 := x0 - x4
			a2 := x2 + x6
			a3 := x2 - x6
			a4 := x1 + x5
			a5 := x1 - x5
			a6 := x3 + x7
			a7 := x3 - x7

			e0 := a0 + a2
			e2 := a0 - a2
			e1 := a1 + complex(imag(a3), -real(a3))
			e3 := a1 + complex(-imag(a3), real(a3))

			o0 := a4 + a6
			o2 := a4 - a6
			o1 := a5 + complex(imag(a7), -real(a7))
			o3 := a5 + complex(-imag(a7), real(a7))

			stage2[base+j] = e0 + o0
			stage2[base+j+32] = e0 - o0
			stage2[base+j+8] = e1 + w1_8*o1
			stage2[base+j+40] = e1 - w1_8*o1
			stage2[base+j+16] = e2 + w2_8*o2
			stage2[base+j+48] = e2 - w2_8*o2
			stage2[base+j+24] = e3 + w3_8*o3
			stage2[base+j+56] = e3 - w3_8*o3
		}
	}

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		tw1 := tw[j]
		tw2 := tw[2*j]
		tw3 := tw[3*j]
		tw4 := tw[4*j]
		tw5 := tw[5*j]
		tw6 := tw[6*j]
		tw7 := tw[7*j]

		x0 := stage2[j]
		x1 := tw1 * stage2[j+64]
		x2 := tw2 * stage2[j+128]
		x3 := tw3 * stage2[j+192]
		x4 := tw4 * stage2[j+256]
		x5 := tw5 * stage2[j+320]
		x6 := tw6 * stage2[j+384]
		x7 := tw7 * stage2[j+448]

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(imag(a3), -real(a3))
		e3 := a1 + complex(-imag(a3), real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(imag(a7), -real(a7))
		o3 := a5 + complex(-imag(a7), real(a7))

		work[j] = e0 + o0
		work[j+256] = e0 - o0
		work[j+64] = e1 + w1_8*o1
		work[j+320] = e1 - w1_8*o1
		work[j+128] = e2 + w2_8*o2
		work[j+384] = e2 - w2_8*o2
		work[j+192] = e3 + w3_8*o3
		work[j+448] = e3 - w3_8*o3
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT512Radix8Complex128 computes a 512-point inverse FFT using
// pure radix-8 Decimation-in-Time (DIT) algorithm for complex128 data.
func inverseDIT512Radix8Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 512

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Internal twiddles for radix-8 butterfly (conjugated for inverse)
	w1_8 := complex(real(tw[64]), -imag(tw[64]))
	w2_8 := complex(real(tw[128]), -imag(tw[128]))
	w3_8 := complex(real(tw[192]), -imag(tw[192]))

	var stage1 [512]complex128

	for base := 0; base < n; base += 8 {
		x0 := s[br[base]]
		x1 := s[br[base+1]]
		x2 := s[br[base+2]]
		x3 := s[br[base+3]]
		x4 := s[br[base+4]]
		x5 := s[br[base+5]]
		x6 := s[br[base+6]]
		x7 := s[br[base+7]]

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(-imag(a3), real(a3))
		e3 := a1 + complex(imag(a3), -real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(-imag(a7), real(a7))
		o3 := a5 + complex(imag(a7), -real(a7))

		stage1[base] = e0 + o0
		stage1[base+4] = e0 - o0
		stage1[base+1] = e1 + w1_8*o1
		stage1[base+5] = e1 - w1_8*o1
		stage1[base+2] = e2 + w2_8*o2
		stage1[base+6] = e2 - w2_8*o2
		stage1[base+3] = e3 + w3_8*o3
		stage1[base+7] = e3 - w3_8*o3
	}

	var stage2 [512]complex128

	for base := 0; base < n; base += 64 {
		for j := range 8 {
			tw1 := complex(real(tw[j*8]), -imag(tw[j*8]))
			tw2 := complex(real(tw[j*16]), -imag(tw[j*16]))
			tw3 := complex(real(tw[j*24]), -imag(tw[j*24]))
			tw4 := complex(real(tw[j*32]), -imag(tw[j*32]))
			tw5 := complex(real(tw[j*40]), -imag(tw[j*40]))
			tw6 := complex(real(tw[j*48]), -imag(tw[j*48]))
			tw7 := complex(real(tw[j*56]), -imag(tw[j*56]))

			x0 := stage1[base+j]
			x1 := tw1 * stage1[base+j+8]
			x2 := tw2 * stage1[base+j+16]
			x3 := tw3 * stage1[base+j+24]
			x4 := tw4 * stage1[base+j+32]
			x5 := tw5 * stage1[base+j+40]
			x6 := tw6 * stage1[base+j+48]
			x7 := tw7 * stage1[base+j+56]

			a0 := x0 + x4
			a1 := x0 - x4
			a2 := x2 + x6
			a3 := x2 - x6
			a4 := x1 + x5
			a5 := x1 - x5
			a6 := x3 + x7
			a7 := x3 - x7

			e0 := a0 + a2
			e2 := a0 - a2
			e1 := a1 + complex(-imag(a3), real(a3))
			e3 := a1 + complex(imag(a3), -real(a3))

			o0 := a4 + a6
			o2 := a4 - a6
			o1 := a5 + complex(-imag(a7), real(a7))
			o3 := a5 + complex(imag(a7), -real(a7))

			stage2[base+j] = e0 + o0
			stage2[base+j+32] = e0 - o0
			stage2[base+j+8] = e1 + w1_8*o1
			stage2[base+j+40] = e1 - w1_8*o1
			stage2[base+j+16] = e2 + w2_8*o2
			stage2[base+j+48] = e2 - w2_8*o2
			stage2[base+j+24] = e3 + w3_8*o3
			stage2[base+j+56] = e3 - w3_8*o3
		}
	}

	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	for j := range 64 {
		tw1 := complex(real(tw[j]), -imag(tw[j]))
		tw2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		tw3 := complex(real(tw[3*j]), -imag(tw[3*j]))
		tw4 := complex(real(tw[4*j]), -imag(tw[4*j]))
		tw5 := complex(real(tw[5*j]), -imag(tw[5*j]))
		tw6 := complex(real(tw[6*j]), -imag(tw[6*j]))
		tw7 := complex(real(tw[7*j]), -imag(tw[7*j]))

		x0 := stage2[j]
		x1 := tw1 * stage2[j+64]
		x2 := tw2 * stage2[j+128]
		x3 := tw3 * stage2[j+192]
		x4 := tw4 * stage2[j+256]
		x5 := tw5 * stage2[j+320]
		x6 := tw6 * stage2[j+384]
		x7 := tw7 * stage2[j+448]

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(-imag(a3), real(a3))
		e3 := a1 + complex(imag(a3), -real(a3))

		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(-imag(a7), real(a7))
		o3 := a5 + complex(imag(a7), -real(a7))

		work[j] = e0 + o0
		work[j+256] = e0 - o0
		work[j+64] = e1 + w1_8*o1
		work[j+320] = e1 - w1_8*o1
		work[j+128] = e2 + w2_8*o2
		work[j+384] = e2 - w2_8*o2
		work[j+192] = e3 + w3_8*o3
		work[j+448] = e3 - w3_8*o3
	}

	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	scale := 1.0 / float64(n)
	for i := range dst[:n] {
		dst[i] *= complex(scale, 0)
	}

	return true
}
