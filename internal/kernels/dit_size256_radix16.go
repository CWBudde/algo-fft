package kernels

// forwardDIT256Radix16Complex64 is an optimized radix-16 DIT FFT for size-256.
// It decomposes the 256-point FFT into a 16x16 matrix (2 stages of radix-16).
func forwardDIT256Radix16Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// Stage 1: 16 parallel Size-16 FFTs on columns.
	for i := 0; i < 256; i += 16 {
		// Load 16 elements
		v0 := s[br[i]]
		v1 := s[br[i+1]]
		v2 := s[br[i+2]]
		v3 := s[br[i+3]]
		v4 := s[br[i+4]]
		v5 := s[br[i+5]]
		v6 := s[br[i+6]]
		v7 := s[br[i+7]]
		v8 := s[br[i+8]]
		v9 := s[br[i+9]]
		v10 := s[br[i+10]]
		v11 := s[br[i+11]]
		v12 := s[br[i+12]]
		v13 := s[br[i+13]]
		v14 := s[br[i+14]]
		v15 := s[br[i+15]]

		// Perform Size-16 FFT (Bit-Reversed Input -> Natural Output)
		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex64(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
		out[i+8] = v8
		out[i+9] = v9
		out[i+10] = v10
		out[i+11] = v11
		out[i+12] = v12
		out[i+13] = v13
		out[i+14] = v14
		out[i+15] = v15
	}

	// Stage 2: 16 parallel Size-16 FFTs on rows.
	for j := range 16 {
		// k=0: Col 0. W^0
		v0 := out[j]

		// k=1: Col 8. W^8j
		v1 := out[j+16] * tw[8*j]

		// k=2: Col 4. W^4j
		v2 := out[j+32] * tw[4*j]

		// k=3: Col 12. W^12j
		v3 := out[j+48] * tw[12*j]

		// k=4: Col 2. W^2j
		v4 := out[j+64] * tw[2*j]

		// k=5: Col 10. W^10j
		v5 := out[j+80] * tw[10*j]

		// k=6: Col 6. W^6j
		v6 := out[j+96] * tw[6*j]

		// k=7: Col 14. W^14j
		v7 := out[j+112] * tw[14*j]

		// k=8: Col 1. W^j
		v8 := out[j+128] * tw[j]

		// k=9: Col 9. W^9j
		v9 := out[j+144] * tw[9*j]

		// k=10: Col 5. W^5j
		v10 := out[j+160] * tw[5*j]

		// k=11: Col 13. W^13j
		v11 := out[j+176] * tw[13*j]

		// k=12: Col 3. W^3j
		v12 := out[j+192] * tw[3*j]

		// k=13: Col 11. W^11j
		v13 := out[j+208] * tw[11*j]

		// k=14: Col 7. W^7j
		v14 := out[j+224] * tw[7*j]

		// k=15: Col 15. W^15j
		v15 := out[j+240] * tw[15*j]

		// Perform Size-16 FFT
		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex64(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		// Write to dst
		dst[j] = v0
		dst[j+16] = v1
		dst[j+32] = v2
		dst[j+48] = v3
		dst[j+64] = v4
		dst[j+80] = v5
		dst[j+96] = v6
		dst[j+112] = v7
		dst[j+128] = v8
		dst[j+144] = v9
		dst[j+160] = v10
		dst[j+176] = v11
		dst[j+192] = v12
		dst[j+208] = v13
		dst[j+224] = v14
		dst[j+240] = v15
	}

	return true
}

// fft16Complex64 computes a size-16 FFT on 16 variables.
func fft16Complex64(
	v0, v1, v2, v3, v4, v5, v6, v7,
	v8, v9, v10, v11, v12, v13, v14, v15 complex64,
) (
	r0, r1, r2, r3, r4, r5, r6, r7,
	r8, r9, r10, r11, r12, r13, r14, r15 complex64,
) {
	const (
		sin1 = float32(0.382683432365089771728459984030398866761)
		cos1 = float32(0.923879532511286756128183189396788286822)
		isq2 = float32(0.707106781186547524400844362104849039284) // 1/sqrt(2)
	)

	// Stage 1
	t0 := v0 + v1
	v1 = v0 - v1
	v0 = t0
	t0 = v2 + v3
	v3 = v2 - v3
	v2 = t0
	t0 = v4 + v5
	v5 = v4 - v5
	v4 = t0
	t0 = v6 + v7
	v7 = v6 - v7
	v6 = t0
	t0 = v8 + v9
	v9 = v8 - v9
	v8 = t0
	t0 = v10 + v11
	v11 = v10 - v11
	v10 = t0
	t0 = v12 + v13
	v13 = v12 - v13
	v12 = t0
	t0 = v14 + v15
	v15 = v14 - v15
	v14 = t0

	// Stage 2
	t0 = v0 + v2
	v2 = v0 - v2
	v0 = t0
	t3 := complex(imag(v3), -real(v3))
	v3 = v1 - t3
	v1 = v1 + t3

	t0 = v4 + v6
	v6 = v4 - v6
	v4 = t0
	t3 = complex(imag(v7), -real(v7))
	v7 = v5 - t3
	v5 = v5 + t3

	t0 = v8 + v10
	v10 = v8 - v10
	v8 = t0
	t3 = complex(imag(v11), -real(v11))
	v11 = v9 - t3
	v9 = v9 + t3

	t0 = v12 + v14
	v14 = v12 - v14
	v12 = t0
	t3 = complex(imag(v15), -real(v15))
	v15 = v13 - t3
	v13 = v13 + t3

	// Stage 3
	t0 = v0 + v4
	v4 = v0 - v4
	v0 = t0
	tr, ti := real(v5), imag(v5)
	t3 = complex(tr*isq2+ti*isq2, ti*isq2-tr*isq2)
	v5 = v1 - t3
	v1 = v1 + t3
	t3 = complex(imag(v6), -real(v6))
	v6 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v7), imag(v7)
	t3 = complex(ti*isq2-tr*isq2, -ti*isq2-tr*isq2)
	v7 = v3 - t3
	v3 = v3 + t3

	t0 = v8 + v12
	v12 = v8 - v12
	v8 = t0
	tr, ti = real(v13), imag(v13)
	t3 = complex(tr*isq2+ti*isq2, ti*isq2-tr*isq2)
	v13 = v9 - t3
	v9 = v9 + t3
	t3 = complex(imag(v14), -real(v14))
	v14 = v10 - t3
	v10 = v10 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(ti*isq2-tr*isq2, -ti*isq2-tr*isq2)
	v15 = v11 - t3
	v11 = v11 + t3

	// Stage 4
	t0 = v0 + v8
	v8 = v0 - v8
	v0 = t0
	tr, ti = real(v9), imag(v9)
	t3 = complex(tr*cos1+ti*sin1, ti*cos1-tr*sin1)
	v9 = v1 - t3
	v1 = v1 + t3
	tr, ti = real(v10), imag(v10)
	t3 = complex(tr*isq2+ti*isq2, ti*isq2-tr*isq2)
	v10 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v11), imag(v11)
	t3 = complex(tr*sin1+ti*cos1, ti*sin1-tr*cos1)
	v11 = v3 - t3
	v3 = v3 + t3
	t3 = complex(imag(v12), -real(v12))
	v12 = v4 - t3
	v4 = v4 + t3
	tr, ti = real(v13), imag(v13)
	t3 = complex(ti*cos1-tr*sin1, -ti*sin1-tr*cos1)
	v13 = v5 - t3
	v5 = v5 + t3
	tr, ti = real(v14), imag(v14)
	t3 = complex(ti*isq2-tr*isq2, -ti*isq2-tr*isq2)
	v14 = v6 - t3
	v6 = v6 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(ti*sin1-tr*cos1, -ti*cos1-tr*sin1)
	v15 = v7 - t3
	v7 = v7 + t3

	r0, r1, r2, r3, r4, r5, r6, r7 = v0, v1, v2, v3, v4, v5, v6, v7
	r8, r9, r10, r11, r12, r13, r14, r15 = v8, v9, v10, v11, v12, v13, v14, v15

	return r0, r1, r2, r3, r4, r5, r6, r7,
		r8, r9, r10, r11, r12, r13, r14, r15
}

// inverseDIT256Radix16Complex64 is an optimized radix-16 DIT inverse FFT for size-256.
func inverseDIT256Radix16Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// Stage 1
	for i := 0; i < 256; i += 16 {
		v0 := s[br[i]]
		v1 := s[br[i+1]]
		v2 := s[br[i+2]]
		v3 := s[br[i+3]]
		v4 := s[br[i+4]]
		v5 := s[br[i+5]]
		v6 := s[br[i+6]]
		v7 := s[br[i+7]]
		v8 := s[br[i+8]]
		v9 := s[br[i+9]]
		v10 := s[br[i+10]]
		v11 := s[br[i+11]]
		v12 := s[br[i+12]]
		v13 := s[br[i+13]]
		v14 := s[br[i+14]]
		v15 := s[br[i+15]]

		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex64Inverse(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
		out[i+8] = v8
		out[i+9] = v9
		out[i+10] = v10
		out[i+11] = v11
		out[i+12] = v12
		out[i+13] = v13
		out[i+14] = v14
		out[i+15] = v15
	}

	// Stage 2
	for j := range 16 {
		v0 := out[j]
		v1 := out[j+16] * complex(real(tw[(8*j)%256]), -imag(tw[(8*j)%256]))
		v2 := out[j+32] * complex(real(tw[(4*j)%256]), -imag(tw[(4*j)%256]))
		v3 := out[j+48] * complex(real(tw[(12*j)%256]), -imag(tw[(12*j)%256]))
		v4 := out[j+64] * complex(real(tw[(2*j)%256]), -imag(tw[(2*j)%256]))
		v5 := out[j+80] * complex(real(tw[(10*j)%256]), -imag(tw[(10*j)%256]))
		v6 := out[j+96] * complex(real(tw[(6*j)%256]), -imag(tw[(6*j)%256]))
		v7 := out[j+112] * complex(real(tw[(14*j)%256]), -imag(tw[(14*j)%256]))
		v8 := out[j+128] * complex(real(tw[j]), -imag(tw[j]))
		v9 := out[j+144] * complex(real(tw[(9*j)%256]), -imag(tw[(9*j)%256]))
		v10 := out[j+160] * complex(real(tw[(5*j)%256]), -imag(tw[(5*j)%256]))
		v11 := out[j+176] * complex(real(tw[(13*j)%256]), -imag(tw[(13*j)%256]))
		v12 := out[j+192] * complex(real(tw[(3*j)%256]), -imag(tw[(3*j)%256]))
		v13 := out[j+208] * complex(real(tw[(11*j)%256]), -imag(tw[(11*j)%256]))
		v14 := out[j+224] * complex(real(tw[(7*j)%256]), -imag(tw[(7*j)%256]))
		v15 := out[j+240] * complex(real(tw[(15*j)%256]), -imag(tw[(15*j)%256]))

		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex64Inverse(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		const scale = 1.0 / 256.0

		dst[j] = v0 * scale
		dst[j+16] = v1 * scale
		dst[j+32] = v2 * scale
		dst[j+48] = v3 * scale
		dst[j+64] = v4 * scale
		dst[j+80] = v5 * scale
		dst[j+96] = v6 * scale
		dst[j+112] = v7 * scale
		dst[j+128] = v8 * scale
		dst[j+144] = v9 * scale
		dst[j+160] = v10 * scale
		dst[j+176] = v11 * scale
		dst[j+192] = v12 * scale
		dst[j+208] = v13 * scale
		dst[j+224] = v14 * scale
		dst[j+240] = v15 * scale
	}

	return true
}

func fft16Complex64Inverse(
	v0, v1, v2, v3, v4, v5, v6, v7,
	v8, v9, v10, v11, v12, v13, v14, v15 complex64,
) (
	r0, r1, r2, r3, r4, r5, r6, r7,
	r8, r9, r10, r11, r12, r13, r14, r15 complex64,
) {
	const (
		P_sin1 = float32(0.382683432365089771728459984030398866761)
		P_cos1 = float32(0.923879532511286756128183189396788286822)
		P_isq2 = float32(0.707106781186547524400844362104849039284)
	)

	// Stage 1
	t0 := v0 + v1
	v1 = v0 - v1
	v0 = t0
	t0 = v2 + v3
	v3 = v2 - v3
	v2 = t0
	t0 = v4 + v5
	v5 = v4 - v5
	v4 = t0
	t0 = v6 + v7
	v7 = v6 - v7
	v6 = t0
	t0 = v8 + v9
	v9 = v8 - v9
	v8 = t0
	t0 = v10 + v11
	v11 = v10 - v11
	v10 = t0
	t0 = v12 + v13
	v13 = v12 - v13
	v12 = t0
	t0 = v14 + v15
	v15 = v14 - v15
	v14 = t0

	// Stage 2. Twiddles W^0=1, W^1=i (Inverse of -i)
	t0 = v0 + v2
	v2 = v0 - v2
	v0 = t0
	t3 := complex(-imag(v3), real(v3))
	v3 = v1 - t3
	v1 = v1 + t3

	t0 = v4 + v6
	v6 = v4 - v6
	v4 = t0
	t3 = complex(-imag(v7), real(v7))
	v7 = v5 - t3
	v5 = v5 + t3

	t0 = v8 + v10
	v10 = v8 - v10
	v8 = t0
	t3 = complex(-imag(v11), real(v11))
	v11 = v9 - t3
	v9 = v9 + t3

	t0 = v12 + v14
	v14 = v12 - v14
	v12 = t0
	t3 = complex(-imag(v15), real(v15))
	v15 = v13 - t3
	v13 = v13 + t3

	// Stage 3. Twiddles W_8^{-k}.
	t0 = v0 + v4
	v4 = v0 - v4
	v0 = t0
	tr, ti := real(v5), imag(v5)
	t3 = complex(tr*P_isq2-ti*P_isq2, tr*P_isq2+ti*P_isq2)
	v5 = v1 - t3
	v1 = v1 + t3
	t3 = complex(-imag(v6), real(v6))
	v6 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v7), imag(v7)
	t3 = complex(-tr*P_isq2-ti*P_isq2, tr*P_isq2-ti*P_isq2)
	v7 = v3 - t3
	v3 = v3 + t3

	t0 = v8 + v12
	v12 = v8 - v12
	v8 = t0
	tr, ti = real(v13), imag(v13)
	t3 = complex(tr*P_isq2-ti*P_isq2, tr*P_isq2+ti*P_isq2)
	v13 = v9 - t3
	v9 = v9 + t3
	t3 = complex(-imag(v14), real(v14))
	v14 = v10 - t3
	v10 = v10 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(-tr*P_isq2-ti*P_isq2, tr*P_isq2-ti*P_isq2)
	v15 = v11 - t3
	v11 = v11 + t3

	// Stage 4. W_16^{-k}.
	t0 = v0 + v8
	v8 = v0 - v8
	v0 = t0
	tr, ti = real(v9), imag(v9)
	t3 = complex(tr*P_cos1-ti*P_sin1, tr*P_sin1+ti*P_cos1)
	v9 = v1 - t3
	v1 = v1 + t3
	tr, ti = real(v10), imag(v10)
	t3 = complex(tr*P_isq2-ti*P_isq2, tr*P_isq2+ti*P_isq2)
	v10 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v11), imag(v11)
	t3 = complex(tr*P_sin1-ti*P_cos1, tr*P_cos1+ti*P_sin1)
	v11 = v3 - t3
	v3 = v3 + t3
	t3 = complex(-imag(v12), real(v12))
	v12 = v4 - t3
	v4 = v4 + t3
	tr, ti = real(v13), imag(v13)
	t3 = complex(tr*(-P_sin1)-ti*P_cos1, tr*P_cos1+ti*(-P_sin1))
	v13 = v5 - t3
	v5 = v5 + t3
	tr, ti = real(v14), imag(v14)
	t3 = complex(tr*(-P_isq2)-ti*P_isq2, tr*P_isq2+ti*(-P_isq2))
	v14 = v6 - t3
	v6 = v6 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(tr*(-P_cos1)-ti*P_sin1, tr*P_sin1+ti*(-P_cos1))
	v15 = v7 - t3
	v7 = v7 + t3

	r0, r1, r2, r3, r4, r5, r6, r7 = v0, v1, v2, v3, v4, v5, v6, v7
	r8, r9, r10, r11, r12, r13, r14, r15 = v8, v9, v10, v11, v12, v13, v14, v15

	return r0, r1, r2, r3, r4, r5, r6, r7,
		r8, r9, r10, r11, r12, r13, r14, r15
}

// forwardDIT256Radix16Complex128 is an optimized radix-16 DIT FFT for size-256 (complex128).
func forwardDIT256Radix16Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// Stage 1
	for i := 0; i < 256; i += 16 {
		v0 := s[br[i]]
		v1 := s[br[i+1]]
		v2 := s[br[i+2]]
		v3 := s[br[i+3]]
		v4 := s[br[i+4]]
		v5 := s[br[i+5]]
		v6 := s[br[i+6]]
		v7 := s[br[i+7]]
		v8 := s[br[i+8]]
		v9 := s[br[i+9]]
		v10 := s[br[i+10]]
		v11 := s[br[i+11]]
		v12 := s[br[i+12]]
		v13 := s[br[i+13]]
		v14 := s[br[i+14]]
		v15 := s[br[i+15]]

		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex128(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
		out[i+8] = v8
		out[i+9] = v9
		out[i+10] = v10
		out[i+11] = v11
		out[i+12] = v12
		out[i+13] = v13
		out[i+14] = v14
		out[i+15] = v15
	}

	// Stage 2
	for j := range 16 {
		v0 := out[j]
		v1 := out[j+16] * tw[(8*j)%256]
		v2 := out[j+32] * tw[(4*j)%256]
		v3 := out[j+48] * tw[(12*j)%256]
		v4 := out[j+64] * tw[(2*j)%256]
		v5 := out[j+80] * tw[(10*j)%256]
		v6 := out[j+96] * tw[(6*j)%256]
		v7 := out[j+112] * tw[(14*j)%256]
		v8 := out[j+128] * tw[j]
		v9 := out[j+144] * tw[(9*j)%256]
		v10 := out[j+160] * tw[(5*j)%256]
		v11 := out[j+176] * tw[(13*j)%256]
		v12 := out[j+192] * tw[(3*j)%256]
		v13 := out[j+208] * tw[(11*j)%256]
		v14 := out[j+224] * tw[(7*j)%256]
		v15 := out[j+240] * tw[(15*j)%256]

		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex128(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		dst[j] = v0
		dst[j+16] = v1
		dst[j+32] = v2
		dst[j+48] = v3
		dst[j+64] = v4
		dst[j+80] = v5
		dst[j+96] = v6
		dst[j+112] = v7
		dst[j+128] = v8
		dst[j+144] = v9
		dst[j+160] = v10
		dst[j+176] = v11
		dst[j+192] = v12
		dst[j+208] = v13
		dst[j+224] = v14
		dst[j+240] = v15
	}

	return true
}

func fft16Complex128(
	v0, v1, v2, v3, v4, v5, v6, v7,
	v8, v9, v10, v11, v12, v13, v14, v15 complex128,
) (
	r0, r1, r2, r3, r4, r5, r6, r7,
	r8, r9, r10, r11, r12, r13, r14, r15 complex128,
) {
	const (
		sin1 = 0.382683432365089771728459984030398866761
		cos1 = 0.923879532511286756128183189396788286822
		isq2 = 0.707106781186547524400844362104849039284
	)

	// Stage 1
	t0 := v0 + v1
	v1 = v0 - v1
	v0 = t0
	t0 = v2 + v3
	v3 = v2 - v3
	v2 = t0
	t0 = v4 + v5
	v5 = v4 - v5
	v4 = t0
	t0 = v6 + v7
	v7 = v6 - v7
	v6 = t0
	t0 = v8 + v9
	v9 = v8 - v9
	v8 = t0
	t0 = v10 + v11
	v11 = v10 - v11
	v10 = t0
	t0 = v12 + v13
	v13 = v12 - v13
	v12 = t0
	t0 = v14 + v15
	v15 = v14 - v15
	v14 = t0

	// Stage 2
	t0 = v0 + v2
	v2 = v0 - v2
	v0 = t0
	t3 := complex(imag(v3), -real(v3))
	v3 = v1 - t3
	v1 = v1 + t3

	t0 = v4 + v6
	v6 = v4 - v6
	v4 = t0
	t3 = complex(imag(v7), -real(v7))
	v7 = v5 - t3
	v5 = v5 + t3

	t0 = v8 + v10
	v10 = v8 - v10
	v8 = t0
	t3 = complex(imag(v11), -real(v11))
	v11 = v9 - t3
	v9 = v9 + t3

	t0 = v12 + v14
	v14 = v12 - v14
	v12 = t0
	t3 = complex(imag(v15), -real(v15))
	v15 = v13 - t3
	v13 = v13 + t3

	// Stage 3
	t0 = v0 + v4
	v4 = v0 - v4
	v0 = t0
	tr, ti := real(v5), imag(v5)
	t3 = complex(tr*isq2+ti*isq2, ti*isq2-tr*isq2)
	v5 = v1 - t3
	v1 = v1 + t3
	t3 = complex(imag(v6), -real(v6))
	v6 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v7), imag(v7)
	t3 = complex(ti*isq2-tr*isq2, -ti*isq2-tr*isq2)
	v7 = v3 - t3
	v3 = v3 + t3

	t0 = v8 + v12
	v12 = v8 - v12
	v8 = t0
	tr, ti = real(v13), imag(v13)
	t3 = complex(tr*isq2+ti*isq2, ti*isq2-tr*isq2)
	v13 = v9 - t3
	v9 = v9 + t3
	t3 = complex(imag(v14), -real(v14))
	v14 = v10 - t3
	v10 = v10 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(ti*isq2-tr*isq2, -ti*isq2-tr*isq2)
	v15 = v11 - t3
	v11 = v11 + t3

	// Stage 4
	t0 = v0 + v8
	v8 = v0 - v8
	v0 = t0
	tr, ti = real(v9), imag(v9)
	t3 = complex(tr*cos1+ti*sin1, ti*cos1-tr*sin1)
	v9 = v1 - t3
	v1 = v1 + t3
	tr, ti = real(v10), imag(v10)
	t3 = complex(tr*isq2+ti*isq2, ti*isq2-tr*isq2)
	v10 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v11), imag(v11)
	t3 = complex(tr*sin1+ti*cos1, ti*sin1-tr*cos1)
	v11 = v3 - t3
	v3 = v3 + t3
	t3 = complex(imag(v12), -real(v12))
	v12 = v4 - t3
	v4 = v4 + t3
	tr, ti = real(v13), imag(v13)
	t3 = complex(ti*cos1-tr*sin1, -ti*sin1-tr*cos1)
	v13 = v5 - t3
	v5 = v5 + t3
	tr, ti = real(v14), imag(v14)
	t3 = complex(ti*isq2-tr*isq2, -ti*isq2-tr*isq2)
	v14 = v6 - t3
	v6 = v6 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(ti*sin1-tr*cos1, -ti*cos1-tr*sin1)
	v15 = v7 - t3
	v7 = v7 + t3

	r0, r1, r2, r3, r4, r5, r6, r7 = v0, v1, v2, v3, v4, v5, v6, v7
	r8, r9, r10, r11, r12, r13, r14, r15 = v8, v9, v10, v11, v12, v13, v14, v15

	return r0, r1, r2, r3, r4, r5, r6, r7,
		r8, r9, r10, r11, r12, r13, r14, r15
}

// inverseDIT256Radix16Complex128 is an optimized radix-16 DIT inverse FFT for size-256 (complex128).
func inverseDIT256Radix16Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// Stage 1
	for i := 0; i < 256; i += 16 {
		v0 := s[br[i]]
		v1 := s[br[i+1]]
		v2 := s[br[i+2]]
		v3 := s[br[i+3]]
		v4 := s[br[i+4]]
		v5 := s[br[i+5]]
		v6 := s[br[i+6]]
		v7 := s[br[i+7]]
		v8 := s[br[i+8]]
		v9 := s[br[i+9]]
		v10 := s[br[i+10]]
		v11 := s[br[i+11]]
		v12 := s[br[i+12]]
		v13 := s[br[i+13]]
		v14 := s[br[i+14]]
		v15 := s[br[i+15]]

		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex128Inverse(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
		out[i+8] = v8
		out[i+9] = v9
		out[i+10] = v10
		out[i+11] = v11
		out[i+12] = v12
		out[i+13] = v13
		out[i+14] = v14
		out[i+15] = v15
	}

	// Stage 2
	for j := range 16 {
		v0 := out[j]
		v1 := out[j+16] * complex(real(tw[(8*j)%256]), -imag(tw[(8*j)%256]))
		v2 := out[j+32] * complex(real(tw[(4*j)%256]), -imag(tw[(4*j)%256]))
		v3 := out[j+48] * complex(real(tw[(12*j)%256]), -imag(tw[(12*j)%256]))
		v4 := out[j+64] * complex(real(tw[(2*j)%256]), -imag(tw[(2*j)%256]))
		v5 := out[j+80] * complex(real(tw[(10*j)%256]), -imag(tw[(10*j)%256]))
		v6 := out[j+96] * complex(real(tw[(6*j)%256]), -imag(tw[(6*j)%256]))
		v7 := out[j+112] * complex(real(tw[(14*j)%256]), -imag(tw[(14*j)%256]))
		v8 := out[j+128] * complex(real(tw[j]), -imag(tw[j]))
		v9 := out[j+144] * complex(real(tw[(9*j)%256]), -imag(tw[(9*j)%256]))
		v10 := out[j+160] * complex(real(tw[(5*j)%256]), -imag(tw[(5*j)%256]))
		v11 := out[j+176] * complex(real(tw[(13*j)%256]), -imag(tw[(13*j)%256]))
		v12 := out[j+192] * complex(real(tw[(3*j)%256]), -imag(tw[(3*j)%256]))
		v13 := out[j+208] * complex(real(tw[(11*j)%256]), -imag(tw[(11*j)%256]))
		v14 := out[j+224] * complex(real(tw[(7*j)%256]), -imag(tw[(7*j)%256]))
		v15 := out[j+240] * complex(real(tw[(15*j)%256]), -imag(tw[(15*j)%256]))

		v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fft16Complex128Inverse(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

		const scale = 1.0 / 256.0

		dst[j] = v0 * scale
		dst[j+16] = v1 * scale
		dst[j+32] = v2 * scale
		dst[j+48] = v3 * scale
		dst[j+64] = v4 * scale
		dst[j+80] = v5 * scale
		dst[j+96] = v6 * scale
		dst[j+112] = v7 * scale
		dst[j+128] = v8 * scale
		dst[j+144] = v9 * scale
		dst[j+160] = v10 * scale
		dst[j+176] = v11 * scale
		dst[j+192] = v12 * scale
		dst[j+208] = v13 * scale
		dst[j+224] = v14 * scale
		dst[j+240] = v15 * scale
	}

	return true
}

func fft16Complex128Inverse(
	v0, v1, v2, v3, v4, v5, v6, v7,
	v8, v9, v10, v11, v12, v13, v14, v15 complex128,
) (
	r0, r1, r2, r3, r4, r5, r6, r7,
	r8, r9, r10, r11, r12, r13, r14, r15 complex128,
) {
	const (
		P_sin1 = 0.382683432365089771728459984030398866761
		P_cos1 = 0.923879532511286756128183189396788286822
		P_isq2 = 0.707106781186547524400844362104849039284
	)

	// Stage 1
	t0 := v0 + v1
	v1 = v0 - v1
	v0 = t0
	t0 = v2 + v3
	v3 = v2 - v3
	v2 = t0
	t0 = v4 + v5
	v5 = v4 - v5
	v4 = t0
	t0 = v6 + v7
	v7 = v6 - v7
	v6 = t0
	t0 = v8 + v9
	v9 = v8 - v9
	v8 = t0
	t0 = v10 + v11
	v11 = v10 - v11
	v10 = t0
	t0 = v12 + v13
	v13 = v12 - v13
	v12 = t0
	t0 = v14 + v15
	v15 = v14 - v15
	v14 = t0

	// Stage 2. Twiddles W^0=1, W^1=i (Inverse of -i)
	t0 = v0 + v2
	v2 = v0 - v2
	v0 = t0
	t3 := complex(-imag(v3), real(v3))
	v3 = v1 - t3
	v1 = v1 + t3

	t0 = v4 + v6
	v6 = v4 - v6
	v4 = t0
	t3 = complex(-imag(v7), real(v7))
	v7 = v5 - t3
	v5 = v5 + t3

	t0 = v8 + v10
	v10 = v8 - v10
	v8 = t0
	t3 = complex(-imag(v11), real(v11))
	v11 = v9 - t3
	v9 = v9 + t3

	t0 = v12 + v14
	v14 = v12 - v14
	v12 = t0
	t3 = complex(-imag(v15), real(v15))
	v15 = v13 - t3
	v13 = v13 + t3

	// Stage 3. Twiddles W_8^{-k}.
	t0 = v0 + v4
	v4 = v0 - v4
	v0 = t0
	tr, ti := real(v5), imag(v5)
	t3 = complex(tr*P_isq2-ti*P_isq2, tr*P_isq2+ti*P_isq2)
	v5 = v1 - t3
	v1 = v1 + t3
	t3 = complex(-imag(v6), real(v6))
	v6 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v7), imag(v7)
	t3 = complex(-tr*P_isq2-ti*P_isq2, tr*P_isq2-ti*P_isq2)
	v7 = v3 - t3
	v3 = v3 + t3

	t0 = v8 + v12
	v12 = v8 - v12
	v8 = t0
	tr, ti = real(v13), imag(v13)
	t3 = complex(tr*P_isq2-ti*P_isq2, tr*P_isq2+ti*P_isq2)
	v13 = v9 - t3
	v9 = v9 + t3
	t3 = complex(-imag(v14), real(v14))
	v14 = v10 - t3
	v10 = v10 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(-tr*P_isq2-ti*P_isq2, tr*P_isq2-ti*P_isq2)
	v15 = v11 - t3
	v11 = v11 + t3

	// Stage 4. W_16^{-k}.
	t0 = v0 + v8
	v8 = v0 - v8
	v0 = t0
	tr, ti = real(v9), imag(v9)
	t3 = complex(tr*P_cos1-ti*P_sin1, tr*P_sin1+ti*P_cos1)
	v9 = v1 - t3
	v1 = v1 + t3
	tr, ti = real(v10), imag(v10)
	t3 = complex(tr*P_isq2-ti*P_isq2, tr*P_isq2+ti*P_isq2)
	v10 = v2 - t3
	v2 = v2 + t3
	tr, ti = real(v11), imag(v11)
	t3 = complex(tr*P_sin1-ti*P_cos1, tr*P_cos1+ti*P_sin1)
	v11 = v3 - t3
	v3 = v3 + t3
	t3 = complex(-imag(v12), real(v12))
	v12 = v4 - t3
	v4 = v4 + t3
	tr, ti = real(v13), imag(v13)
	t3 = complex(tr*(-P_sin1)-ti*P_cos1, tr*P_cos1+ti*(-P_sin1))
	v13 = v5 - t3
	v5 = v5 + t3
	tr, ti = real(v14), imag(v14)
	t3 = complex(tr*(-P_isq2)-ti*P_isq2, tr*P_isq2+ti*(-P_isq2))
	v14 = v6 - t3
	v6 = v6 + t3
	tr, ti = real(v15), imag(v15)
	t3 = complex(tr*(-P_cos1)-ti*P_sin1, tr*P_sin1+ti*(-P_cos1))
	v15 = v7 - t3
	v7 = v7 + t3

	r0, r1, r2, r3, r4, r5, r6, r7 = v0, v1, v2, v3, v4, v5, v6, v7
	r8, r9, r10, r11, r12, r13, r14, r15 = v8, v9, v10, v11, v12, v13, v14, v15

	return r0, r1, r2, r3, r4, r5, r6, r7,
		r8, r9, r10, r11, r12, r13, r14, r15
}
