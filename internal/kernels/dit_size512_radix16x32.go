package kernels

// forwardDIT512Mixed16x32Complex64 is an optimized 16x32 mixed-radix DIT FFT for size-512.
// Uses six-step FFT algorithm: n = 16*n2 + n1, k = 32*k1 + k2
// Stage 1: 16 FFT-32s on columns, Stage 2: 32 FFT-16s on rows
func forwardDIT512Mixed16x32Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 512
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// 16-point bit-reversal indices for DIT FFT-16
	const (
		br16_0  = 0
		br16_1  = 8
		br16_2  = 4
		br16_3  = 12
		br16_4  = 2
		br16_5  = 10
		br16_6  = 6
		br16_7  = 14
		br16_8  = 1
		br16_9  = 9
		br16_10 = 5
		br16_11 = 13
		br16_12 = 3
		br16_13 = 11
		br16_14 = 7
		br16_15 = 15
	)

	// Stage 1: 16 FFT-32s on columns.
	// Since we don't have a correct fft32Complex64, we implement it as two layers of FFT-16
	// using the Cooley-Tukey decomposition: FFT-32 = 2 * FFT-16 with twiddle factors.
	// 32 = 2 * 16, so for each column we do:
	// - Two FFT-16s on even and odd indexed elements
	// - Combine with W_32 twiddle factors
	for n1 := 0; n1 < 16; n1++ {
		// Load 32 elements from column n1 (stride 16)
		// Even indices: n2 = 0, 2, 4, ..., 30 -> positions 0, 32, 64, ..., 480 -> 16*n2+n1
		// Odd indices:  n2 = 1, 3, 5, ..., 31 -> positions 16, 48, 80, ..., 496 -> 16*n2+n1
		e0 := s[16*0+n1]
		e1 := s[16*2+n1]
		e2 := s[16*4+n1]
		e3 := s[16*6+n1]
		e4 := s[16*8+n1]
		e5 := s[16*10+n1]
		e6 := s[16*12+n1]
		e7 := s[16*14+n1]
		e8 := s[16*16+n1]
		e9 := s[16*18+n1]
		e10 := s[16*20+n1]
		e11 := s[16*22+n1]
		e12 := s[16*24+n1]
		e13 := s[16*26+n1]
		e14 := s[16*28+n1]
		e15 := s[16*30+n1]

		o0 := s[16*1+n1]
		o1 := s[16*3+n1]
		o2 := s[16*5+n1]
		o3 := s[16*7+n1]
		o4 := s[16*9+n1]
		o5 := s[16*11+n1]
		o6 := s[16*13+n1]
		o7 := s[16*15+n1]
		o8 := s[16*17+n1]
		o9 := s[16*19+n1]
		o10 := s[16*21+n1]
		o11 := s[16*23+n1]
		o12 := s[16*25+n1]
		o13 := s[16*27+n1]
		o14 := s[16*29+n1]
		o15 := s[16*31+n1]

		// FFT-16 on even elements (bit-reversed input)
		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex64(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// FFT-16 on odd elements (bit-reversed input)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex64(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with W_32 twiddle factors: W_32^k = tw[k*16] (from 512-point table)
		// Y[k] = E[k] + W_32^k * O[k], Y[k+16] = E[k] - W_32^k * O[k] for k=0..15
		// Then apply inter-stage twiddle W_512^{k2*n1} = tw[k2*n1]

		// k2 = 0: W_32^0 = 1
		out[0*16+n1] = (E0 + O0) * tw[0]
		out[16*16+n1] = (E0 - O0) * tw[16*n1]

		// k2 = 1: W_32^1 = tw[16]
		t1 := O1 * tw[16]
		out[1*16+n1] = (E1 + t1) * tw[1*n1]
		out[17*16+n1] = (E1 - t1) * tw[17*n1]

		// k2 = 2: W_32^2 = tw[32]
		t2 := O2 * tw[32]
		out[2*16+n1] = (E2 + t2) * tw[2*n1]
		out[18*16+n1] = (E2 - t2) * tw[18*n1]

		// k2 = 3: W_32^3 = tw[48]
		t3 := O3 * tw[48]
		out[3*16+n1] = (E3 + t3) * tw[3*n1]
		out[19*16+n1] = (E3 - t3) * tw[19*n1]

		// k2 = 4: W_32^4 = tw[64]
		t4 := O4 * tw[64]
		out[4*16+n1] = (E4 + t4) * tw[4*n1]
		out[20*16+n1] = (E4 - t4) * tw[20*n1]

		// k2 = 5: W_32^5 = tw[80]
		t5 := O5 * tw[80]
		out[5*16+n1] = (E5 + t5) * tw[5*n1]
		out[21*16+n1] = (E5 - t5) * tw[21*n1]

		// k2 = 6: W_32^6 = tw[96]
		t6 := O6 * tw[96]
		out[6*16+n1] = (E6 + t6) * tw[6*n1]
		out[22*16+n1] = (E6 - t6) * tw[22*n1]

		// k2 = 7: W_32^7 = tw[112]
		t7 := O7 * tw[112]
		out[7*16+n1] = (E7 + t7) * tw[7*n1]
		out[23*16+n1] = (E7 - t7) * tw[23*n1]

		// k2 = 8: W_32^8 = tw[128]
		t8 := O8 * tw[128]
		out[8*16+n1] = (E8 + t8) * tw[8*n1]
		out[24*16+n1] = (E8 - t8) * tw[24*n1]

		// k2 = 9: W_32^9 = tw[144]
		t9 := O9 * tw[144]
		out[9*16+n1] = (E9 + t9) * tw[9*n1]
		out[25*16+n1] = (E9 - t9) * tw[25*n1]

		// k2 = 10: W_32^10 = tw[160]
		t10 := O10 * tw[160]
		out[10*16+n1] = (E10 + t10) * tw[10*n1]
		out[26*16+n1] = (E10 - t10) * tw[26*n1]

		// k2 = 11: W_32^11 = tw[176]
		t11 := O11 * tw[176]
		out[11*16+n1] = (E11 + t11) * tw[11*n1]
		out[27*16+n1] = (E11 - t11) * tw[27*n1]

		// k2 = 12: W_32^12 = tw[192]
		t12 := O12 * tw[192]
		out[12*16+n1] = (E12 + t12) * tw[12*n1]
		out[28*16+n1] = (E12 - t12) * tw[28*n1]

		// k2 = 13: W_32^13 = tw[208]
		t13 := O13 * tw[208]
		out[13*16+n1] = (E13 + t13) * tw[13*n1]
		out[29*16+n1] = (E13 - t13) * tw[29*n1]

		// k2 = 14: W_32^14 = tw[224]
		t14 := O14 * tw[224]
		out[14*16+n1] = (E14 + t14) * tw[14*n1]
		out[30*16+n1] = (E14 - t14) * tw[30*n1]

		// k2 = 15: W_32^15 = tw[240]
		t15 := O15 * tw[240]
		out[15*16+n1] = (E15 + t15) * tw[15*n1]
		out[31*16+n1] = (E15 - t15) * tw[31*n1]
	}

	// Stage 2: 32 FFT-16s on rows using DIT fft16Complex64 (bit-reversed input -> natural output).
	for k2 := 0; k2 < 32; k2++ {
		base := k2 * 16
		// Load 16 inputs in bit-reversed order for DIT FFT-16
		z0 := out[base+br16_0]
		z1 := out[base+br16_1]
		z2 := out[base+br16_2]
		z3 := out[base+br16_3]
		z4 := out[base+br16_4]
		z5 := out[base+br16_5]
		z6 := out[base+br16_6]
		z7 := out[base+br16_7]
		z8 := out[base+br16_8]
		z9 := out[base+br16_9]
		z10 := out[base+br16_10]
		z11 := out[base+br16_11]
		z12 := out[base+br16_12]
		z13 := out[base+br16_13]
		z14 := out[base+br16_14]
		z15 := out[base+br16_15]

		// Perform 16-point FFT (bit-reversed input -> natural output)
		r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 :=
			fft16Complex64(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15)

		// Store to output: dst[32*k1 + k2] for k1 = 0..15
		dst[32*0+k2] = r0
		dst[32*1+k2] = r1
		dst[32*2+k2] = r2
		dst[32*3+k2] = r3
		dst[32*4+k2] = r4
		dst[32*5+k2] = r5
		dst[32*6+k2] = r6
		dst[32*7+k2] = r7
		dst[32*8+k2] = r8
		dst[32*9+k2] = r9
		dst[32*10+k2] = r10
		dst[32*11+k2] = r11
		dst[32*12+k2] = r12
		dst[32*13+k2] = r13
		dst[32*14+k2] = r14
		dst[32*15+k2] = r15
	}

	return true
}

func inverseDIT512Mixed16x32Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 512
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// 16-point bit-reversal indices for DIT IFFT-16
	const (
		br16_0  = 0
		br16_1  = 8
		br16_2  = 4
		br16_3  = 12
		br16_4  = 2
		br16_5  = 10
		br16_6  = 6
		br16_7  = 14
		br16_8  = 1
		br16_9  = 9
		br16_10 = 5
		br16_11 = 13
		br16_12 = 3
		br16_13 = 11
		br16_14 = 7
		br16_15 = 15
	)

	// Stage 1: 32 IFFT-16s on rows using DIT ifft16 (bit-reversed input -> natural output).
	// Input X[32*k1 + k2], output Y[k2, n1]
	for k2 := 0; k2 < 32; k2++ {
		// Load 16 inputs from row k2 (stride 32) in bit-reversed order
		z0 := s[32*br16_0+k2]
		z1 := s[32*br16_1+k2]
		z2 := s[32*br16_2+k2]
		z3 := s[32*br16_3+k2]
		z4 := s[32*br16_4+k2]
		z5 := s[32*br16_5+k2]
		z6 := s[32*br16_6+k2]
		z7 := s[32*br16_7+k2]
		z8 := s[32*br16_8+k2]
		z9 := s[32*br16_9+k2]
		z10 := s[32*br16_10+k2]
		z11 := s[32*br16_11+k2]
		z12 := s[32*br16_12+k2]
		z13 := s[32*br16_13+k2]
		z14 := s[32*br16_14+k2]
		z15 := s[32*br16_15+k2]

		// Perform IFFT-16 (bit-reversed input -> natural output)
		r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 :=
			fft16Complex64Inverse(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15)

		// Store with conjugate inter-stage twiddle W_512^{-k2*n1} = conj(tw[k2*n1])
		base := k2 * 16
		out[base+0] = r0 * conj(tw[k2*0])
		out[base+1] = r1 * conj(tw[k2*1])
		out[base+2] = r2 * conj(tw[k2*2])
		out[base+3] = r3 * conj(tw[k2*3])
		out[base+4] = r4 * conj(tw[k2*4])
		out[base+5] = r5 * conj(tw[k2*5])
		out[base+6] = r6 * conj(tw[k2*6])
		out[base+7] = r7 * conj(tw[k2*7])
		out[base+8] = r8 * conj(tw[k2*8])
		out[base+9] = r9 * conj(tw[k2*9])
		out[base+10] = r10 * conj(tw[k2*10])
		out[base+11] = r11 * conj(tw[k2*11])
		out[base+12] = r12 * conj(tw[k2*12])
		out[base+13] = r13 * conj(tw[k2*13])
		out[base+14] = r14 * conj(tw[k2*14])
		out[base+15] = r15 * conj(tw[k2*15])
	}

	// Stage 2: 16 IFFT-32s on columns.
	// IFFT-32 = 2 * IFFT-16 combined with conjugate W_32 twiddle factors.
	const scale = float32(1.0 / 512.0)
	for n1 := 0; n1 < 16; n1++ {
		// Load 32 elements from column n1 (stride 16)
		// Even indices: k2 = 0, 2, 4, ..., 30
		// Odd indices:  k2 = 1, 3, 5, ..., 31
		e0 := out[16*0+n1]
		e1 := out[16*2+n1]
		e2 := out[16*4+n1]
		e3 := out[16*6+n1]
		e4 := out[16*8+n1]
		e5 := out[16*10+n1]
		e6 := out[16*12+n1]
		e7 := out[16*14+n1]
		e8 := out[16*16+n1]
		e9 := out[16*18+n1]
		e10 := out[16*20+n1]
		e11 := out[16*22+n1]
		e12 := out[16*24+n1]
		e13 := out[16*26+n1]
		e14 := out[16*28+n1]
		e15 := out[16*30+n1]

		o0 := out[16*1+n1]
		o1 := out[16*3+n1]
		o2 := out[16*5+n1]
		o3 := out[16*7+n1]
		o4 := out[16*9+n1]
		o5 := out[16*11+n1]
		o6 := out[16*13+n1]
		o7 := out[16*15+n1]
		o8 := out[16*17+n1]
		o9 := out[16*19+n1]
		o10 := out[16*21+n1]
		o11 := out[16*23+n1]
		o12 := out[16*25+n1]
		o13 := out[16*27+n1]
		o14 := out[16*29+n1]
		o15 := out[16*31+n1]

		// IFFT-16 on even elements (bit-reversed input)
		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex64Inverse(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// IFFT-16 on odd elements (bit-reversed input)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex64Inverse(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with conjugate W_32 twiddle factors: W_32^{-k} = conj(tw[k*16])
		// x[n2] = E[n2] + W_32^{-n2} * O[n2], x[n2+16] = E[n2] - W_32^{-n2} * O[n2] for n2=0..15
		// Apply 1/512 scaling

		// n2 = 0: W_32^0 = 1
		dst[16*0+n1] = (E0 + O0) * complex(scale, 0)
		dst[16*16+n1] = (E0 - O0) * complex(scale, 0)

		// n2 = 1: W_32^{-1} = conj(tw[16])
		t1 := O1 * conj(tw[16])
		dst[16*1+n1] = (E1 + t1) * complex(scale, 0)
		dst[16*17+n1] = (E1 - t1) * complex(scale, 0)

		// n2 = 2: W_32^{-2} = conj(tw[32])
		t2 := O2 * conj(tw[32])
		dst[16*2+n1] = (E2 + t2) * complex(scale, 0)
		dst[16*18+n1] = (E2 - t2) * complex(scale, 0)

		// n2 = 3: W_32^{-3} = conj(tw[48])
		t3 := O3 * conj(tw[48])
		dst[16*3+n1] = (E3 + t3) * complex(scale, 0)
		dst[16*19+n1] = (E3 - t3) * complex(scale, 0)

		// n2 = 4: W_32^{-4} = conj(tw[64])
		t4 := O4 * conj(tw[64])
		dst[16*4+n1] = (E4 + t4) * complex(scale, 0)
		dst[16*20+n1] = (E4 - t4) * complex(scale, 0)

		// n2 = 5: W_32^{-5} = conj(tw[80])
		t5 := O5 * conj(tw[80])
		dst[16*5+n1] = (E5 + t5) * complex(scale, 0)
		dst[16*21+n1] = (E5 - t5) * complex(scale, 0)

		// n2 = 6: W_32^{-6} = conj(tw[96])
		t6 := O6 * conj(tw[96])
		dst[16*6+n1] = (E6 + t6) * complex(scale, 0)
		dst[16*22+n1] = (E6 - t6) * complex(scale, 0)

		// n2 = 7: W_32^{-7} = conj(tw[112])
		t7 := O7 * conj(tw[112])
		dst[16*7+n1] = (E7 + t7) * complex(scale, 0)
		dst[16*23+n1] = (E7 - t7) * complex(scale, 0)

		// n2 = 8: W_32^{-8} = conj(tw[128])
		t8 := O8 * conj(tw[128])
		dst[16*8+n1] = (E8 + t8) * complex(scale, 0)
		dst[16*24+n1] = (E8 - t8) * complex(scale, 0)

		// n2 = 9: W_32^{-9} = conj(tw[144])
		t9 := O9 * conj(tw[144])
		dst[16*9+n1] = (E9 + t9) * complex(scale, 0)
		dst[16*25+n1] = (E9 - t9) * complex(scale, 0)

		// n2 = 10: W_32^{-10} = conj(tw[160])
		t10 := O10 * conj(tw[160])
		dst[16*10+n1] = (E10 + t10) * complex(scale, 0)
		dst[16*26+n1] = (E10 - t10) * complex(scale, 0)

		// n2 = 11: W_32^{-11} = conj(tw[176])
		t11 := O11 * conj(tw[176])
		dst[16*11+n1] = (E11 + t11) * complex(scale, 0)
		dst[16*27+n1] = (E11 - t11) * complex(scale, 0)

		// n2 = 12: W_32^{-12} = conj(tw[192])
		t12 := O12 * conj(tw[192])
		dst[16*12+n1] = (E12 + t12) * complex(scale, 0)
		dst[16*28+n1] = (E12 - t12) * complex(scale, 0)

		// n2 = 13: W_32^{-13} = conj(tw[208])
		t13 := O13 * conj(tw[208])
		dst[16*13+n1] = (E13 + t13) * complex(scale, 0)
		dst[16*29+n1] = (E13 - t13) * complex(scale, 0)

		// n2 = 14: W_32^{-14} = conj(tw[224])
		t14 := O14 * conj(tw[224])
		dst[16*14+n1] = (E14 + t14) * complex(scale, 0)
		dst[16*30+n1] = (E14 - t14) * complex(scale, 0)

		// n2 = 15: W_32^{-15} = conj(tw[240])
		t15 := O15 * conj(tw[240])
		dst[16*15+n1] = (E15 + t15) * complex(scale, 0)
		dst[16*31+n1] = (E15 - t15) * complex(scale, 0)
	}

	return true
}

func forwardDIT512Mixed16x32Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 512
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// 16-point bit-reversal indices for DIT FFT-16
	const (
		br16_0  = 0
		br16_1  = 8
		br16_2  = 4
		br16_3  = 12
		br16_4  = 2
		br16_5  = 10
		br16_6  = 6
		br16_7  = 14
		br16_8  = 1
		br16_9  = 9
		br16_10 = 5
		br16_11 = 13
		br16_12 = 3
		br16_13 = 11
		br16_14 = 7
		br16_15 = 15
	)

	// Stage 1: 16 FFT-32s on columns using Cooley-Tukey decomposition.
	// FFT-32 = 2 * FFT-16 with twiddle factors.
	for n1 := 0; n1 < 16; n1++ {
		// Load even and odd indexed elements from column
		e0 := s[16*0+n1]
		e1 := s[16*2+n1]
		e2 := s[16*4+n1]
		e3 := s[16*6+n1]
		e4 := s[16*8+n1]
		e5 := s[16*10+n1]
		e6 := s[16*12+n1]
		e7 := s[16*14+n1]
		e8 := s[16*16+n1]
		e9 := s[16*18+n1]
		e10 := s[16*20+n1]
		e11 := s[16*22+n1]
		e12 := s[16*24+n1]
		e13 := s[16*26+n1]
		e14 := s[16*28+n1]
		e15 := s[16*30+n1]

		o0 := s[16*1+n1]
		o1 := s[16*3+n1]
		o2 := s[16*5+n1]
		o3 := s[16*7+n1]
		o4 := s[16*9+n1]
		o5 := s[16*11+n1]
		o6 := s[16*13+n1]
		o7 := s[16*15+n1]
		o8 := s[16*17+n1]
		o9 := s[16*19+n1]
		o10 := s[16*21+n1]
		o11 := s[16*23+n1]
		o12 := s[16*25+n1]
		o13 := s[16*27+n1]
		o14 := s[16*29+n1]
		o15 := s[16*31+n1]

		// FFT-16 on even elements (bit-reversed input)
		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex128(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// FFT-16 on odd elements (bit-reversed input)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex128(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with W_32 twiddle factors and inter-stage twiddles
		out[0*16+n1] = (E0 + O0) * tw[0]
		out[16*16+n1] = (E0 - O0) * tw[16*n1]

		t1 := O1 * tw[16]
		out[1*16+n1] = (E1 + t1) * tw[1*n1]
		out[17*16+n1] = (E1 - t1) * tw[17*n1]

		t2 := O2 * tw[32]
		out[2*16+n1] = (E2 + t2) * tw[2*n1]
		out[18*16+n1] = (E2 - t2) * tw[18*n1]

		t3 := O3 * tw[48]
		out[3*16+n1] = (E3 + t3) * tw[3*n1]
		out[19*16+n1] = (E3 - t3) * tw[19*n1]

		t4 := O4 * tw[64]
		out[4*16+n1] = (E4 + t4) * tw[4*n1]
		out[20*16+n1] = (E4 - t4) * tw[20*n1]

		t5 := O5 * tw[80]
		out[5*16+n1] = (E5 + t5) * tw[5*n1]
		out[21*16+n1] = (E5 - t5) * tw[21*n1]

		t6 := O6 * tw[96]
		out[6*16+n1] = (E6 + t6) * tw[6*n1]
		out[22*16+n1] = (E6 - t6) * tw[22*n1]

		t7 := O7 * tw[112]
		out[7*16+n1] = (E7 + t7) * tw[7*n1]
		out[23*16+n1] = (E7 - t7) * tw[23*n1]

		t8 := O8 * tw[128]
		out[8*16+n1] = (E8 + t8) * tw[8*n1]
		out[24*16+n1] = (E8 - t8) * tw[24*n1]

		t9 := O9 * tw[144]
		out[9*16+n1] = (E9 + t9) * tw[9*n1]
		out[25*16+n1] = (E9 - t9) * tw[25*n1]

		t10 := O10 * tw[160]
		out[10*16+n1] = (E10 + t10) * tw[10*n1]
		out[26*16+n1] = (E10 - t10) * tw[26*n1]

		t11 := O11 * tw[176]
		out[11*16+n1] = (E11 + t11) * tw[11*n1]
		out[27*16+n1] = (E11 - t11) * tw[27*n1]

		t12 := O12 * tw[192]
		out[12*16+n1] = (E12 + t12) * tw[12*n1]
		out[28*16+n1] = (E12 - t12) * tw[28*n1]

		t13 := O13 * tw[208]
		out[13*16+n1] = (E13 + t13) * tw[13*n1]
		out[29*16+n1] = (E13 - t13) * tw[29*n1]

		t14 := O14 * tw[224]
		out[14*16+n1] = (E14 + t14) * tw[14*n1]
		out[30*16+n1] = (E14 - t14) * tw[30*n1]

		t15 := O15 * tw[240]
		out[15*16+n1] = (E15 + t15) * tw[15*n1]
		out[31*16+n1] = (E15 - t15) * tw[31*n1]
	}

	// Stage 2: 32 FFT-16s on rows
	for k2 := 0; k2 < 32; k2++ {
		base := k2 * 16
		z0 := out[base+br16_0]
		z1 := out[base+br16_1]
		z2 := out[base+br16_2]
		z3 := out[base+br16_3]
		z4 := out[base+br16_4]
		z5 := out[base+br16_5]
		z6 := out[base+br16_6]
		z7 := out[base+br16_7]
		z8 := out[base+br16_8]
		z9 := out[base+br16_9]
		z10 := out[base+br16_10]
		z11 := out[base+br16_11]
		z12 := out[base+br16_12]
		z13 := out[base+br16_13]
		z14 := out[base+br16_14]
		z15 := out[base+br16_15]

		r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 :=
			fft16Complex128(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15)

		dst[32*0+k2] = r0
		dst[32*1+k2] = r1
		dst[32*2+k2] = r2
		dst[32*3+k2] = r3
		dst[32*4+k2] = r4
		dst[32*5+k2] = r5
		dst[32*6+k2] = r6
		dst[32*7+k2] = r7
		dst[32*8+k2] = r8
		dst[32*9+k2] = r9
		dst[32*10+k2] = r10
		dst[32*11+k2] = r11
		dst[32*12+k2] = r12
		dst[32*13+k2] = r13
		dst[32*14+k2] = r14
		dst[32*15+k2] = r15
	}

	return true
}

func inverseDIT512Mixed16x32Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 512
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// 16-point bit-reversal indices for DIT IFFT-16
	const (
		br16_0  = 0
		br16_1  = 8
		br16_2  = 4
		br16_3  = 12
		br16_4  = 2
		br16_5  = 10
		br16_6  = 6
		br16_7  = 14
		br16_8  = 1
		br16_9  = 9
		br16_10 = 5
		br16_11 = 13
		br16_12 = 3
		br16_13 = 11
		br16_14 = 7
		br16_15 = 15
	)

	// Stage 1: 32 IFFT-16s on rows
	for k2 := 0; k2 < 32; k2++ {
		z0 := s[32*br16_0+k2]
		z1 := s[32*br16_1+k2]
		z2 := s[32*br16_2+k2]
		z3 := s[32*br16_3+k2]
		z4 := s[32*br16_4+k2]
		z5 := s[32*br16_5+k2]
		z6 := s[32*br16_6+k2]
		z7 := s[32*br16_7+k2]
		z8 := s[32*br16_8+k2]
		z9 := s[32*br16_9+k2]
		z10 := s[32*br16_10+k2]
		z11 := s[32*br16_11+k2]
		z12 := s[32*br16_12+k2]
		z13 := s[32*br16_13+k2]
		z14 := s[32*br16_14+k2]
		z15 := s[32*br16_15+k2]

		r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 :=
			fft16Complex128Inverse(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15)

		base := k2 * 16
		out[base+0] = r0 * conj(tw[k2*0])
		out[base+1] = r1 * conj(tw[k2*1])
		out[base+2] = r2 * conj(tw[k2*2])
		out[base+3] = r3 * conj(tw[k2*3])
		out[base+4] = r4 * conj(tw[k2*4])
		out[base+5] = r5 * conj(tw[k2*5])
		out[base+6] = r6 * conj(tw[k2*6])
		out[base+7] = r7 * conj(tw[k2*7])
		out[base+8] = r8 * conj(tw[k2*8])
		out[base+9] = r9 * conj(tw[k2*9])
		out[base+10] = r10 * conj(tw[k2*10])
		out[base+11] = r11 * conj(tw[k2*11])
		out[base+12] = r12 * conj(tw[k2*12])
		out[base+13] = r13 * conj(tw[k2*13])
		out[base+14] = r14 * conj(tw[k2*14])
		out[base+15] = r15 * conj(tw[k2*15])
	}

	// Stage 2: 16 IFFT-32s on columns using Cooley-Tukey decomposition
	const scale = 1.0 / 512.0
	for n1 := 0; n1 < 16; n1++ {
		e0 := out[16*0+n1]
		e1 := out[16*2+n1]
		e2 := out[16*4+n1]
		e3 := out[16*6+n1]
		e4 := out[16*8+n1]
		e5 := out[16*10+n1]
		e6 := out[16*12+n1]
		e7 := out[16*14+n1]
		e8 := out[16*16+n1]
		e9 := out[16*18+n1]
		e10 := out[16*20+n1]
		e11 := out[16*22+n1]
		e12 := out[16*24+n1]
		e13 := out[16*26+n1]
		e14 := out[16*28+n1]
		e15 := out[16*30+n1]

		o0 := out[16*1+n1]
		o1 := out[16*3+n1]
		o2 := out[16*5+n1]
		o3 := out[16*7+n1]
		o4 := out[16*9+n1]
		o5 := out[16*11+n1]
		o6 := out[16*13+n1]
		o7 := out[16*15+n1]
		o8 := out[16*17+n1]
		o9 := out[16*19+n1]
		o10 := out[16*21+n1]
		o11 := out[16*23+n1]
		o12 := out[16*25+n1]
		o13 := out[16*27+n1]
		o14 := out[16*29+n1]
		o15 := out[16*31+n1]

		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex128Inverse(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex128Inverse(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		dst[16*0+n1] = (E0 + O0) * complex(scale, 0)
		dst[16*16+n1] = (E0 - O0) * complex(scale, 0)

		t1 := O1 * conj(tw[16])
		dst[16*1+n1] = (E1 + t1) * complex(scale, 0)
		dst[16*17+n1] = (E1 - t1) * complex(scale, 0)

		t2 := O2 * conj(tw[32])
		dst[16*2+n1] = (E2 + t2) * complex(scale, 0)
		dst[16*18+n1] = (E2 - t2) * complex(scale, 0)

		t3 := O3 * conj(tw[48])
		dst[16*3+n1] = (E3 + t3) * complex(scale, 0)
		dst[16*19+n1] = (E3 - t3) * complex(scale, 0)

		t4 := O4 * conj(tw[64])
		dst[16*4+n1] = (E4 + t4) * complex(scale, 0)
		dst[16*20+n1] = (E4 - t4) * complex(scale, 0)

		t5 := O5 * conj(tw[80])
		dst[16*5+n1] = (E5 + t5) * complex(scale, 0)
		dst[16*21+n1] = (E5 - t5) * complex(scale, 0)

		t6 := O6 * conj(tw[96])
		dst[16*6+n1] = (E6 + t6) * complex(scale, 0)
		dst[16*22+n1] = (E6 - t6) * complex(scale, 0)

		t7 := O7 * conj(tw[112])
		dst[16*7+n1] = (E7 + t7) * complex(scale, 0)
		dst[16*23+n1] = (E7 - t7) * complex(scale, 0)

		t8 := O8 * conj(tw[128])
		dst[16*8+n1] = (E8 + t8) * complex(scale, 0)
		dst[16*24+n1] = (E8 - t8) * complex(scale, 0)

		t9 := O9 * conj(tw[144])
		dst[16*9+n1] = (E9 + t9) * complex(scale, 0)
		dst[16*25+n1] = (E9 - t9) * complex(scale, 0)

		t10 := O10 * conj(tw[160])
		dst[16*10+n1] = (E10 + t10) * complex(scale, 0)
		dst[16*26+n1] = (E10 - t10) * complex(scale, 0)

		t11 := O11 * conj(tw[176])
		dst[16*11+n1] = (E11 + t11) * complex(scale, 0)
		dst[16*27+n1] = (E11 - t11) * complex(scale, 0)

		t12 := O12 * conj(tw[192])
		dst[16*12+n1] = (E12 + t12) * complex(scale, 0)
		dst[16*28+n1] = (E12 - t12) * complex(scale, 0)

		t13 := O13 * conj(tw[208])
		dst[16*13+n1] = (E13 + t13) * complex(scale, 0)
		dst[16*29+n1] = (E13 - t13) * complex(scale, 0)

		t14 := O14 * conj(tw[224])
		dst[16*14+n1] = (E14 + t14) * complex(scale, 0)
		dst[16*30+n1] = (E14 - t14) * complex(scale, 0)

		t15 := O15 * conj(tw[240])
		dst[16*15+n1] = (E15 + t15) * complex(scale, 0)
		dst[16*31+n1] = (E15 - t15) * complex(scale, 0)
	}

	return true
}
