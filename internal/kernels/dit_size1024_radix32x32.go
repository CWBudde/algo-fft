package kernels

// stage1ForwardDIT1024Radix32x32Complex64 computes the 32 parallel FFT-32s on columns
// for a 32x32 mixed-radix decomposition. Output is written to out in column-major order.
func stage1ForwardDIT1024Radix32x32Complex64(out, src, tw []complex64) {
	for n1 := range 32 {
		// Load 32 elements from column n1 (stride 32), split into even/odd indices.
		e0 := src[32*0+n1]
		e1 := src[32*2+n1]
		e2 := src[32*4+n1]
		e3 := src[32*6+n1]
		e4 := src[32*8+n1]
		e5 := src[32*10+n1]
		e6 := src[32*12+n1]
		e7 := src[32*14+n1]
		e8 := src[32*16+n1]
		e9 := src[32*18+n1]
		e10 := src[32*20+n1]
		e11 := src[32*22+n1]
		e12 := src[32*24+n1]
		e13 := src[32*26+n1]
		e14 := src[32*28+n1]
		e15 := src[32*30+n1]

		o0 := src[32*1+n1]
		o1 := src[32*3+n1]
		o2 := src[32*5+n1]
		o3 := src[32*7+n1]
		o4 := src[32*9+n1]
		o5 := src[32*11+n1]
		o6 := src[32*13+n1]
		o7 := src[32*15+n1]
		o8 := src[32*17+n1]
		o9 := src[32*19+n1]
		o10 := src[32*21+n1]
		o11 := src[32*23+n1]
		o12 := src[32*25+n1]
		o13 := src[32*27+n1]
		o14 := src[32*29+n1]
		o15 := src[32*31+n1]

		// FFT-16 on even elements (bit-reversed input).
		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex64(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// FFT-16 on odd elements (bit-reversed input).
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex64(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with W_32 twiddle factors and apply inter-stage twiddles W_1024^{k2*n1}.
		out[0*32+n1] = (E0 + O0) * tw[0]
		out[16*32+n1] = (E0 - O0) * tw[16*n1]

		t1 := O1 * tw[32]
		out[1*32+n1] = (E1 + t1) * tw[1*n1]
		out[17*32+n1] = (E1 - t1) * tw[17*n1]

		t2 := O2 * tw[64]
		out[2*32+n1] = (E2 + t2) * tw[2*n1]
		out[18*32+n1] = (E2 - t2) * tw[18*n1]

		t3 := O3 * tw[96]
		out[3*32+n1] = (E3 + t3) * tw[3*n1]
		out[19*32+n1] = (E3 - t3) * tw[19*n1]

		t4 := O4 * tw[128]
		out[4*32+n1] = (E4 + t4) * tw[4*n1]
		out[20*32+n1] = (E4 - t4) * tw[20*n1]

		t5 := O5 * tw[160]
		out[5*32+n1] = (E5 + t5) * tw[5*n1]
		out[21*32+n1] = (E5 - t5) * tw[21*n1]

		t6 := O6 * tw[192]
		out[6*32+n1] = (E6 + t6) * tw[6*n1]
		out[22*32+n1] = (E6 - t6) * tw[22*n1]

		t7 := O7 * tw[224]
		out[7*32+n1] = (E7 + t7) * tw[7*n1]
		out[23*32+n1] = (E7 - t7) * tw[23*n1]

		t8 := O8 * tw[256]
		out[8*32+n1] = (E8 + t8) * tw[8*n1]
		out[24*32+n1] = (E8 - t8) * tw[24*n1]

		t9 := O9 * tw[288]
		out[9*32+n1] = (E9 + t9) * tw[9*n1]
		out[25*32+n1] = (E9 - t9) * tw[25*n1]

		t10 := O10 * tw[320]
		out[10*32+n1] = (E10 + t10) * tw[10*n1]
		out[26*32+n1] = (E10 - t10) * tw[26*n1]

		t11 := O11 * tw[352]
		out[11*32+n1] = (E11 + t11) * tw[11*n1]
		out[27*32+n1] = (E11 - t11) * tw[27*n1]

		t12 := O12 * tw[384]
		out[12*32+n1] = (E12 + t12) * tw[12*n1]
		out[28*32+n1] = (E12 - t12) * tw[28*n1]

		t13 := O13 * tw[416]
		out[13*32+n1] = (E13 + t13) * tw[13*n1]
		out[29*32+n1] = (E13 - t13) * tw[29*n1]

		t14 := O14 * tw[448]
		out[14*32+n1] = (E14 + t14) * tw[14*n1]
		out[30*32+n1] = (E14 - t14) * tw[30*n1]

		t15 := O15 * tw[480]
		out[15*32+n1] = (E15 + t15) * tw[15*n1]
		out[31*32+n1] = (E15 - t15) * tw[31*n1]
	}
}

// stage1ForwardDIT1024Radix32x32Complex128 computes the 32 parallel FFT-32s on columns
// for a 32x32 mixed-radix decomposition. Output is written to out in column-major order.
func stage1ForwardDIT1024Radix32x32Complex128(out, src, tw []complex128) {
	for n1 := range 32 {
		// Load 32 elements from column n1 (stride 32), split into even/odd indices.
		e0 := src[32*0+n1]
		e1 := src[32*2+n1]
		e2 := src[32*4+n1]
		e3 := src[32*6+n1]
		e4 := src[32*8+n1]
		e5 := src[32*10+n1]
		e6 := src[32*12+n1]
		e7 := src[32*14+n1]
		e8 := src[32*16+n1]
		e9 := src[32*18+n1]
		e10 := src[32*20+n1]
		e11 := src[32*22+n1]
		e12 := src[32*24+n1]
		e13 := src[32*26+n1]
		e14 := src[32*28+n1]
		e15 := src[32*30+n1]

		o0 := src[32*1+n1]
		o1 := src[32*3+n1]
		o2 := src[32*5+n1]
		o3 := src[32*7+n1]
		o4 := src[32*9+n1]
		o5 := src[32*11+n1]
		o6 := src[32*13+n1]
		o7 := src[32*15+n1]
		o8 := src[32*17+n1]
		o9 := src[32*19+n1]
		o10 := src[32*21+n1]
		o11 := src[32*23+n1]
		o12 := src[32*25+n1]
		o13 := src[32*27+n1]
		o14 := src[32*29+n1]
		o15 := src[32*31+n1]

		// FFT-16 on even elements (bit-reversed input).
		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex128(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// FFT-16 on odd elements (bit-reversed input).
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex128(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with W_32 twiddle factors and apply inter-stage twiddles W_1024^{k2*n1}.
		out[0*32+n1] = (E0 + O0) * tw[0]
		out[16*32+n1] = (E0 - O0) * tw[16*n1]

		t1 := O1 * tw[32]
		out[1*32+n1] = (E1 + t1) * tw[1*n1]
		out[17*32+n1] = (E1 - t1) * tw[17*n1]

		t2 := O2 * tw[64]
		out[2*32+n1] = (E2 + t2) * tw[2*n1]
		out[18*32+n1] = (E2 - t2) * tw[18*n1]

		t3 := O3 * tw[96]
		out[3*32+n1] = (E3 + t3) * tw[3*n1]
		out[19*32+n1] = (E3 - t3) * tw[19*n1]

		t4 := O4 * tw[128]
		out[4*32+n1] = (E4 + t4) * tw[4*n1]
		out[20*32+n1] = (E4 - t4) * tw[20*n1]

		t5 := O5 * tw[160]
		out[5*32+n1] = (E5 + t5) * tw[5*n1]
		out[21*32+n1] = (E5 - t5) * tw[21*n1]

		t6 := O6 * tw[192]
		out[6*32+n1] = (E6 + t6) * tw[6*n1]
		out[22*32+n1] = (E6 - t6) * tw[22*n1]

		t7 := O7 * tw[224]
		out[7*32+n1] = (E7 + t7) * tw[7*n1]
		out[23*32+n1] = (E7 - t7) * tw[23*n1]

		t8 := O8 * tw[256]
		out[8*32+n1] = (E8 + t8) * tw[8*n1]
		out[24*32+n1] = (E8 - t8) * tw[24*n1]

		t9 := O9 * tw[288]
		out[9*32+n1] = (E9 + t9) * tw[9*n1]
		out[25*32+n1] = (E9 - t9) * tw[25*n1]

		t10 := O10 * tw[320]
		out[10*32+n1] = (E10 + t10) * tw[10*n1]
		out[26*32+n1] = (E10 - t10) * tw[26*n1]

		t11 := O11 * tw[352]
		out[11*32+n1] = (E11 + t11) * tw[11*n1]
		out[27*32+n1] = (E11 - t11) * tw[27*n1]

		t12 := O12 * tw[384]
		out[12*32+n1] = (E12 + t12) * tw[12*n1]
		out[28*32+n1] = (E12 - t12) * tw[28*n1]

		t13 := O13 * tw[416]
		out[13*32+n1] = (E13 + t13) * tw[13*n1]
		out[29*32+n1] = (E13 - t13) * tw[29*n1]

		t14 := O14 * tw[448]
		out[14*32+n1] = (E14 + t14) * tw[14*n1]
		out[30*32+n1] = (E14 - t14) * tw[30*n1]

		t15 := O15 * tw[480]
		out[15*32+n1] = (E15 + t15) * tw[15*n1]
		out[31*32+n1] = (E15 - t15) * tw[31*n1]
	}
}

// forwardDIT1024Mixed32x32Complex64 computes a 1024-point forward FFT using a 32x32
// mixed-radix decomposition: 32 FFT-32s on columns, twiddle multiply, then 32 FFT-32s on rows.
func forwardDIT1024Mixed32x32Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 1024
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	stage1ForwardDIT1024Radix32x32Complex64(out, s, tw)

	// Stage 2: 32 FFT-32s on rows (k2 fixed, FFT over n1).
	for k2 := range 32 {
		base := k2 * 32

		e0 := out[base+0]
		e1 := out[base+2]
		e2 := out[base+4]
		e3 := out[base+6]
		e4 := out[base+8]
		e5 := out[base+10]
		e6 := out[base+12]
		e7 := out[base+14]
		e8 := out[base+16]
		e9 := out[base+18]
		e10 := out[base+20]
		e11 := out[base+22]
		e12 := out[base+24]
		e13 := out[base+26]
		e14 := out[base+28]
		e15 := out[base+30]

		o0 := out[base+1]
		o1 := out[base+3]
		o2 := out[base+5]
		o3 := out[base+7]
		o4 := out[base+9]
		o5 := out[base+11]
		o6 := out[base+13]
		o7 := out[base+15]
		o8 := out[base+17]
		o9 := out[base+19]
		o10 := out[base+21]
		o11 := out[base+23]
		o12 := out[base+25]
		o13 := out[base+27]
		o14 := out[base+29]
		o15 := out[base+31]

		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex64(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex64(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		r0 := E0 + O0
		r16 := E0 - O0

		t1 := O1 * tw[32]
		r1 := E1 + t1
		r17 := E1 - t1

		t2 := O2 * tw[64]
		r2 := E2 + t2
		r18 := E2 - t2

		t3 := O3 * tw[96]
		r3 := E3 + t3
		r19 := E3 - t3

		t4 := O4 * tw[128]
		r4 := E4 + t4
		r20 := E4 - t4

		t5 := O5 * tw[160]
		r5 := E5 + t5
		r21 := E5 - t5

		t6 := O6 * tw[192]
		r6 := E6 + t6
		r22 := E6 - t6

		t7 := O7 * tw[224]
		r7 := E7 + t7
		r23 := E7 - t7

		t8 := O8 * tw[256]
		r8 := E8 + t8
		r24 := E8 - t8

		t9 := O9 * tw[288]
		r9 := E9 + t9
		r25 := E9 - t9

		t10 := O10 * tw[320]
		r10 := E10 + t10
		r26 := E10 - t10

		t11 := O11 * tw[352]
		r11 := E11 + t11
		r27 := E11 - t11

		t12 := O12 * tw[384]
		r12 := E12 + t12
		r28 := E12 - t12

		t13 := O13 * tw[416]
		r13 := E13 + t13
		r29 := E13 - t13

		t14 := O14 * tw[448]
		r14 := E14 + t14
		r30 := E14 - t14

		t15 := O15 * tw[480]
		r15 := E15 + t15
		r31 := E15 - t15

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
		dst[32*16+k2] = r16
		dst[32*17+k2] = r17
		dst[32*18+k2] = r18
		dst[32*19+k2] = r19
		dst[32*20+k2] = r20
		dst[32*21+k2] = r21
		dst[32*22+k2] = r22
		dst[32*23+k2] = r23
		dst[32*24+k2] = r24
		dst[32*25+k2] = r25
		dst[32*26+k2] = r26
		dst[32*27+k2] = r27
		dst[32*28+k2] = r28
		dst[32*29+k2] = r29
		dst[32*30+k2] = r30
		dst[32*31+k2] = r31
	}

	return true
}

// inverseDIT1024Mixed32x32Complex64 computes a 1024-point inverse FFT using a 32x32
// mixed-radix decomposition with final 1/1024 scaling.
func inverseDIT1024Mixed32x32Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 1024
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// Stage 1: 32 IFFT-32s on rows, then apply inter-stage twiddles.
	for k2 := range 32 {
		// Load first 16 values for even FFT-16
		z0 := s[32*0+k2]
		z1 := s[32*16+k2]
		z2 := s[32*8+k2]
		z3 := s[32*24+k2]
		z4 := s[32*4+k2]
		z5 := s[32*20+k2]
		z6 := s[32*12+k2]
		z7 := s[32*28+k2]
		z8 := s[32*2+k2]
		z9 := s[32*18+k2]
		z10 := s[32*10+k2]
		z11 := s[32*26+k2]
		z12 := s[32*6+k2]
		z13 := s[32*22+k2]
		z14 := s[32*14+k2]
		z15 := s[32*30+k2]

		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 := fft16Complex64Inverse(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15)

		// Reuse variables for odd FFT-16
		z0 = s[32*1+k2]
		z1 = s[32*17+k2]
		z2 = s[32*9+k2]
		z3 = s[32*25+k2]
		z4 = s[32*5+k2]
		z5 = s[32*21+k2]
		z6 = s[32*13+k2]
		z7 = s[32*29+k2]
		z8 = s[32*3+k2]
		z9 = s[32*19+k2]
		z10 = s[32*11+k2]
		z11 = s[32*27+k2]
		z12 = s[32*7+k2]
		z13 = s[32*23+k2]
		z14 = s[32*15+k2]
		z15 = s[32*31+k2]

		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 := fft16Complex64Inverse(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15)

		r0 := e0 + o0
		r16 := e0 - o0

		t1 := o1 * conj(tw[32])
		r1 := e1 + t1
		r17 := e1 - t1

		t2 := o2 * conj(tw[64])
		r2 := e2 + t2
		r18 := e2 - t2

		t3 := o3 * conj(tw[96])
		r3 := e3 + t3
		r19 := e3 - t3

		t4 := o4 * conj(tw[128])
		r4 := e4 + t4
		r20 := e4 - t4

		t5 := o5 * conj(tw[160])
		r5 := e5 + t5
		r21 := e5 - t5

		t6 := o6 * conj(tw[192])
		r6 := e6 + t6
		r22 := e6 - t6

		t7 := o7 * conj(tw[224])
		r7 := e7 + t7
		r23 := e7 - t7

		t8 := o8 * conj(tw[256])
		r8 := e8 + t8
		r24 := e8 - t8

		t9 := o9 * conj(tw[288])
		r9 := e9 + t9
		r25 := e9 - t9

		t10 := o10 * conj(tw[320])
		r10 := e10 + t10
		r26 := e10 - t10

		t11 := o11 * conj(tw[352])
		r11 := e11 + t11
		r27 := e11 - t11

		t12 := o12 * conj(tw[384])
		r12 := e12 + t12
		r28 := e12 - t12

		t13 := o13 * conj(tw[416])
		r13 := e13 + t13
		r29 := e13 - t13

		t14 := o14 * conj(tw[448])
		r14 := e14 + t14
		r30 := e14 - t14

		t15 := o15 * conj(tw[480])
		r15 := e15 + t15
		r31 := e15 - t15

		base := k2 * 32
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
		out[base+16] = r16 * conj(tw[k2*16])
		out[base+17] = r17 * conj(tw[k2*17])
		out[base+18] = r18 * conj(tw[k2*18])
		out[base+19] = r19 * conj(tw[k2*19])
		out[base+20] = r20 * conj(tw[k2*20])
		out[base+21] = r21 * conj(tw[k2*21])
		out[base+22] = r22 * conj(tw[k2*22])
		out[base+23] = r23 * conj(tw[k2*23])
		out[base+24] = r24 * conj(tw[k2*24])
		out[base+25] = r25 * conj(tw[k2*25])
		out[base+26] = r26 * conj(tw[k2*26])
		out[base+27] = r27 * conj(tw[k2*27])
		out[base+28] = r28 * conj(tw[k2*28])
		out[base+29] = r29 * conj(tw[k2*29])
		out[base+30] = r30 * conj(tw[k2*30])
		out[base+31] = r31 * conj(tw[k2*31])
	}

	// Stage 2: 32 IFFT-32s on columns, scale by 1/1024.
	const scale = float32(1.0 / 1024.0)

	for n1 := range 32 {
		e0 := out[32*0+n1]
		e1 := out[32*2+n1]
		e2 := out[32*4+n1]
		e3 := out[32*6+n1]
		e4 := out[32*8+n1]
		e5 := out[32*10+n1]
		e6 := out[32*12+n1]
		e7 := out[32*14+n1]
		e8 := out[32*16+n1]
		e9 := out[32*18+n1]
		e10 := out[32*20+n1]
		e11 := out[32*22+n1]
		e12 := out[32*24+n1]
		e13 := out[32*26+n1]
		e14 := out[32*28+n1]
		e15 := out[32*30+n1]

		o0 := out[32*1+n1]
		o1 := out[32*3+n1]
		o2 := out[32*5+n1]
		o3 := out[32*7+n1]
		o4 := out[32*9+n1]
		o5 := out[32*11+n1]
		o6 := out[32*13+n1]
		o7 := out[32*15+n1]
		o8 := out[32*17+n1]
		o9 := out[32*19+n1]
		o10 := out[32*21+n1]
		o11 := out[32*23+n1]
		o12 := out[32*25+n1]
		o13 := out[32*27+n1]
		o14 := out[32*29+n1]
		o15 := out[32*31+n1]

		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex64Inverse(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex64Inverse(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		x0 := E0 + O0
		x16 := E0 - O0

		t1 := O1 * conj(tw[32])
		x1 := E1 + t1
		x17 := E1 - t1

		t2 := O2 * conj(tw[64])
		x2 := E2 + t2
		x18 := E2 - t2

		t3 := O3 * conj(tw[96])
		x3 := E3 + t3
		x19 := E3 - t3

		t4 := O4 * conj(tw[128])
		x4 := E4 + t4
		x20 := E4 - t4

		t5 := O5 * conj(tw[160])
		x5 := E5 + t5
		x21 := E5 - t5

		t6 := O6 * conj(tw[192])
		x6 := E6 + t6
		x22 := E6 - t6

		t7 := O7 * conj(tw[224])
		x7 := E7 + t7
		x23 := E7 - t7

		t8 := O8 * conj(tw[256])
		x8 := E8 + t8
		x24 := E8 - t8

		t9 := O9 * conj(tw[288])
		x9 := E9 + t9
		x25 := E9 - t9

		t10 := O10 * conj(tw[320])
		x10 := E10 + t10
		x26 := E10 - t10

		t11 := O11 * conj(tw[352])
		x11 := E11 + t11
		x27 := E11 - t11

		t12 := O12 * conj(tw[384])
		x12 := E12 + t12
		x28 := E12 - t12

		t13 := O13 * conj(tw[416])
		x13 := E13 + t13
		x29 := E13 - t13

		t14 := O14 * conj(tw[448])
		x14 := E14 + t14
		x30 := E14 - t14

		t15 := O15 * conj(tw[480])
		x15 := E15 + t15
		x31 := E15 - t15

		scaleComplex := complex(scale, 0)
		dst[32*0+n1] = x0 * scaleComplex
		dst[32*1+n1] = x1 * scaleComplex
		dst[32*2+n1] = x2 * scaleComplex
		dst[32*3+n1] = x3 * scaleComplex
		dst[32*4+n1] = x4 * scaleComplex
		dst[32*5+n1] = x5 * scaleComplex
		dst[32*6+n1] = x6 * scaleComplex
		dst[32*7+n1] = x7 * scaleComplex
		dst[32*8+n1] = x8 * scaleComplex
		dst[32*9+n1] = x9 * scaleComplex
		dst[32*10+n1] = x10 * scaleComplex
		dst[32*11+n1] = x11 * scaleComplex
		dst[32*12+n1] = x12 * scaleComplex
		dst[32*13+n1] = x13 * scaleComplex
		dst[32*14+n1] = x14 * scaleComplex
		dst[32*15+n1] = x15 * scaleComplex
		dst[32*16+n1] = x16 * scaleComplex
		dst[32*17+n1] = x17 * scaleComplex
		dst[32*18+n1] = x18 * scaleComplex
		dst[32*19+n1] = x19 * scaleComplex
		dst[32*20+n1] = x20 * scaleComplex
		dst[32*21+n1] = x21 * scaleComplex
		dst[32*22+n1] = x22 * scaleComplex
		dst[32*23+n1] = x23 * scaleComplex
		dst[32*24+n1] = x24 * scaleComplex
		dst[32*25+n1] = x25 * scaleComplex
		dst[32*26+n1] = x26 * scaleComplex
		dst[32*27+n1] = x27 * scaleComplex
		dst[32*28+n1] = x28 * scaleComplex
		dst[32*29+n1] = x29 * scaleComplex
		dst[32*30+n1] = x30 * scaleComplex
		dst[32*31+n1] = x31 * scaleComplex
	}

	return true
}

// forwardDIT1024Mixed32x32Complex128 computes a 1024-point forward FFT using a 32x32
// mixed-radix decomposition: 32 FFT-32s on columns, twiddle multiply, then 32 FFT-32s on rows.
func forwardDIT1024Mixed32x32Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 1024
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	stage1ForwardDIT1024Radix32x32Complex128(out, s, tw)

	// Stage 2: 32 FFT-32s on rows (k2 fixed, FFT over n1).
	for k2 := range 32 {
		base := k2 * 32

		e0 := out[base+0]
		e1 := out[base+2]
		e2 := out[base+4]
		e3 := out[base+6]
		e4 := out[base+8]
		e5 := out[base+10]
		e6 := out[base+12]
		e7 := out[base+14]
		e8 := out[base+16]
		e9 := out[base+18]
		e10 := out[base+20]
		e11 := out[base+22]
		e12 := out[base+24]
		e13 := out[base+26]
		e14 := out[base+28]
		e15 := out[base+30]

		o0 := out[base+1]
		o1 := out[base+3]
		o2 := out[base+5]
		o3 := out[base+7]
		o4 := out[base+9]
		o5 := out[base+11]
		o6 := out[base+13]
		o7 := out[base+15]
		o8 := out[base+17]
		o9 := out[base+19]
		o10 := out[base+21]
		o11 := out[base+23]
		o12 := out[base+25]
		o13 := out[base+27]
		o14 := out[base+29]
		o15 := out[base+31]

		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex128(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex128(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		r0 := E0 + O0
		r16 := E0 - O0

		t1 := O1 * tw[32]
		r1 := E1 + t1
		r17 := E1 - t1

		t2 := O2 * tw[64]
		r2 := E2 + t2
		r18 := E2 - t2

		t3 := O3 * tw[96]
		r3 := E3 + t3
		r19 := E3 - t3

		t4 := O4 * tw[128]
		r4 := E4 + t4
		r20 := E4 - t4

		t5 := O5 * tw[160]
		r5 := E5 + t5
		r21 := E5 - t5

		t6 := O6 * tw[192]
		r6 := E6 + t6
		r22 := E6 - t6

		t7 := O7 * tw[224]
		r7 := E7 + t7
		r23 := E7 - t7

		t8 := O8 * tw[256]
		r8 := E8 + t8
		r24 := E8 - t8

		t9 := O9 * tw[288]
		r9 := E9 + t9
		r25 := E9 - t9

		t10 := O10 * tw[320]
		r10 := E10 + t10
		r26 := E10 - t10

		t11 := O11 * tw[352]
		r11 := E11 + t11
		r27 := E11 - t11

		t12 := O12 * tw[384]
		r12 := E12 + t12
		r28 := E12 - t12

		t13 := O13 * tw[416]
		r13 := E13 + t13
		r29 := E13 - t13

		t14 := O14 * tw[448]
		r14 := E14 + t14
		r30 := E14 - t14

		t15 := O15 * tw[480]
		r15 := E15 + t15
		r31 := E15 - t15

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
		dst[32*16+k2] = r16
		dst[32*17+k2] = r17
		dst[32*18+k2] = r18
		dst[32*19+k2] = r19
		dst[32*20+k2] = r20
		dst[32*21+k2] = r21
		dst[32*22+k2] = r22
		dst[32*23+k2] = r23
		dst[32*24+k2] = r24
		dst[32*25+k2] = r25
		dst[32*26+k2] = r26
		dst[32*27+k2] = r27
		dst[32*28+k2] = r28
		dst[32*29+k2] = r29
		dst[32*30+k2] = r30
		dst[32*31+k2] = r31
	}

	return true
}

// inverseDIT1024Mixed32x32Complex128 computes a 1024-point inverse FFT using a 32x32
// mixed-radix decomposition with final 1/1024 scaling.
func inverseDIT1024Mixed32x32Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 1024
	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	s := src[:n]
	tw := twiddle[:n]
	out := scratch[:n]

	// Stage 1: 32 IFFT-32s on rows, then apply inter-stage twiddles.
	for k2 := range 32 {
		z0 := s[32*0+k2]
		z1 := s[32*1+k2]
		z2 := s[32*2+k2]
		z3 := s[32*3+k2]
		z4 := s[32*4+k2]
		z5 := s[32*5+k2]
		z6 := s[32*6+k2]
		z7 := s[32*7+k2]
		z8 := s[32*8+k2]
		z9 := s[32*9+k2]
		z10 := s[32*10+k2]
		z11 := s[32*11+k2]
		z12 := s[32*12+k2]
		z13 := s[32*13+k2]
		z14 := s[32*14+k2]
		z15 := s[32*15+k2]
		z16 := s[32*16+k2]
		z17 := s[32*17+k2]
		z18 := s[32*18+k2]
		z19 := s[32*19+k2]
		z20 := s[32*20+k2]
		z21 := s[32*21+k2]
		z22 := s[32*22+k2]
		z23 := s[32*23+k2]
		z24 := s[32*24+k2]
		z25 := s[32*25+k2]
		z26 := s[32*26+k2]
		z27 := s[32*27+k2]
		z28 := s[32*28+k2]
		z29 := s[32*29+k2]
		z30 := s[32*30+k2]
		z31 := s[32*31+k2]

		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 := fft16Complex128Inverse(z0, z16, z8, z24, z4, z20, z12, z28, z2, z18, z10, z26, z6, z22, z14, z30)
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 := fft16Complex128Inverse(z1, z17, z9, z25, z5, z21, z13, z29, z3, z19, z11, z27, z7, z23, z15, z31)

		r0 := e0 + o0
		r16 := e0 - o0

		t1 := o1 * conj(tw[32])
		r1 := e1 + t1
		r17 := e1 - t1

		t2 := o2 * conj(tw[64])
		r2 := e2 + t2
		r18 := e2 - t2

		t3 := o3 * conj(tw[96])
		r3 := e3 + t3
		r19 := e3 - t3

		t4 := o4 * conj(tw[128])
		r4 := e4 + t4
		r20 := e4 - t4

		t5 := o5 * conj(tw[160])
		r5 := e5 + t5
		r21 := e5 - t5

		t6 := o6 * conj(tw[192])
		r6 := e6 + t6
		r22 := e6 - t6

		t7 := o7 * conj(tw[224])
		r7 := e7 + t7
		r23 := e7 - t7

		t8 := o8 * conj(tw[256])
		r8 := e8 + t8
		r24 := e8 - t8

		t9 := o9 * conj(tw[288])
		r9 := e9 + t9
		r25 := e9 - t9

		t10 := o10 * conj(tw[320])
		r10 := e10 + t10
		r26 := e10 - t10

		t11 := o11 * conj(tw[352])
		r11 := e11 + t11
		r27 := e11 - t11

		t12 := o12 * conj(tw[384])
		r12 := e12 + t12
		r28 := e12 - t12

		t13 := o13 * conj(tw[416])
		r13 := e13 + t13
		r29 := e13 - t13

		t14 := o14 * conj(tw[448])
		r14 := e14 + t14
		r30 := e14 - t14

		t15 := o15 * conj(tw[480])
		r15 := e15 + t15
		r31 := e15 - t15

		base := k2 * 32
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
		out[base+16] = r16 * conj(tw[k2*16])
		out[base+17] = r17 * conj(tw[k2*17])
		out[base+18] = r18 * conj(tw[k2*18])
		out[base+19] = r19 * conj(tw[k2*19])
		out[base+20] = r20 * conj(tw[k2*20])
		out[base+21] = r21 * conj(tw[k2*21])
		out[base+22] = r22 * conj(tw[k2*22])
		out[base+23] = r23 * conj(tw[k2*23])
		out[base+24] = r24 * conj(tw[k2*24])
		out[base+25] = r25 * conj(tw[k2*25])
		out[base+26] = r26 * conj(tw[k2*26])
		out[base+27] = r27 * conj(tw[k2*27])
		out[base+28] = r28 * conj(tw[k2*28])
		out[base+29] = r29 * conj(tw[k2*29])
		out[base+30] = r30 * conj(tw[k2*30])
		out[base+31] = r31 * conj(tw[k2*31])
	}

	// Stage 2: 32 IFFT-32s on columns, scale by 1/1024.
	const scale = 1.0 / 1024.0

	for n1 := range 32 {
		e0 := out[32*0+n1]
		e1 := out[32*2+n1]
		e2 := out[32*4+n1]
		e3 := out[32*6+n1]
		e4 := out[32*8+n1]
		e5 := out[32*10+n1]
		e6 := out[32*12+n1]
		e7 := out[32*14+n1]
		e8 := out[32*16+n1]
		e9 := out[32*18+n1]
		e10 := out[32*20+n1]
		e11 := out[32*22+n1]
		e12 := out[32*24+n1]
		e13 := out[32*26+n1]
		e14 := out[32*28+n1]
		e15 := out[32*30+n1]

		o0 := out[32*1+n1]
		o1 := out[32*3+n1]
		o2 := out[32*5+n1]
		o3 := out[32*7+n1]
		o4 := out[32*9+n1]
		o5 := out[32*11+n1]
		o6 := out[32*13+n1]
		o7 := out[32*15+n1]
		o8 := out[32*17+n1]
		o9 := out[32*19+n1]
		o10 := out[32*21+n1]
		o11 := out[32*23+n1]
		o12 := out[32*25+n1]
		o13 := out[32*27+n1]
		o14 := out[32*29+n1]
		o15 := out[32*31+n1]

		E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15 := fft16Complex128Inverse(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 := fft16Complex128Inverse(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		x0 := E0 + O0
		x16 := E0 - O0

		t1 := O1 * conj(tw[32])
		x1 := E1 + t1
		x17 := E1 - t1

		t2 := O2 * conj(tw[64])
		x2 := E2 + t2
		x18 := E2 - t2

		t3 := O3 * conj(tw[96])
		x3 := E3 + t3
		x19 := E3 - t3

		t4 := O4 * conj(tw[128])
		x4 := E4 + t4
		x20 := E4 - t4

		t5 := O5 * conj(tw[160])
		x5 := E5 + t5
		x21 := E5 - t5

		t6 := O6 * conj(tw[192])
		x6 := E6 + t6
		x22 := E6 - t6

		t7 := O7 * conj(tw[224])
		x7 := E7 + t7
		x23 := E7 - t7

		t8 := O8 * conj(tw[256])
		x8 := E8 + t8
		x24 := E8 - t8

		t9 := O9 * conj(tw[288])
		x9 := E9 + t9
		x25 := E9 - t9

		t10 := O10 * conj(tw[320])
		x10 := E10 + t10
		x26 := E10 - t10

		t11 := O11 * conj(tw[352])
		x11 := E11 + t11
		x27 := E11 - t11

		t12 := O12 * conj(tw[384])
		x12 := E12 + t12
		x28 := E12 - t12

		t13 := O13 * conj(tw[416])
		x13 := E13 + t13
		x29 := E13 - t13

		t14 := O14 * conj(tw[448])
		x14 := E14 + t14
		x30 := E14 - t14

		t15 := O15 * conj(tw[480])
		x15 := E15 + t15
		x31 := E15 - t15

		dst[32*0+n1] = x0 * scale
		dst[32*1+n1] = x1 * scale
		dst[32*2+n1] = x2 * scale
		dst[32*3+n1] = x3 * scale
		dst[32*4+n1] = x4 * scale
		dst[32*5+n1] = x5 * scale
		dst[32*6+n1] = x6 * scale
		dst[32*7+n1] = x7 * scale
		dst[32*8+n1] = x8 * scale
		dst[32*9+n1] = x9 * scale
		dst[32*10+n1] = x10 * scale
		dst[32*11+n1] = x11 * scale
		dst[32*12+n1] = x12 * scale
		dst[32*13+n1] = x13 * scale
		dst[32*14+n1] = x14 * scale
		dst[32*15+n1] = x15 * scale
		dst[32*16+n1] = x16 * scale
		dst[32*17+n1] = x17 * scale
		dst[32*18+n1] = x18 * scale
		dst[32*19+n1] = x19 * scale
		dst[32*20+n1] = x20 * scale
		dst[32*21+n1] = x21 * scale
		dst[32*22+n1] = x22 * scale
		dst[32*23+n1] = x23 * scale
		dst[32*24+n1] = x24 * scale
		dst[32*25+n1] = x25 * scale
		dst[32*26+n1] = x26 * scale
		dst[32*27+n1] = x27 * scale
		dst[32*28+n1] = x28 * scale
		dst[32*29+n1] = x29 * scale
		dst[32*30+n1] = x30 * scale
		dst[32*31+n1] = x31 * scale
	}

	return true
}
