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
		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = fft16Complex64(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// FFT-16 on odd elements (bit-reversed input).
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = fft16Complex64(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with W_32 twiddle factors and apply inter-stage twiddles W_1024^{k2*n1}.
		out[0*32+n1] = (e0 + o0) * tw[0]
		out[16*32+n1] = (e0 - o0) * tw[16*n1]

		t1 := o1 * tw[32]
		out[1*32+n1] = (e1 + t1) * tw[1*n1]
		out[17*32+n1] = (e1 - t1) * tw[17*n1]

		t2 := o2 * tw[64]
		out[2*32+n1] = (e2 + t2) * tw[2*n1]
		out[18*32+n1] = (e2 - t2) * tw[18*n1]

		t3 := o3 * tw[96]
		out[3*32+n1] = (e3 + t3) * tw[3*n1]
		out[19*32+n1] = (e3 - t3) * tw[19*n1]

		t4 := o4 * tw[128]
		out[4*32+n1] = (e4 + t4) * tw[4*n1]
		out[20*32+n1] = (e4 - t4) * tw[20*n1]

		t5 := o5 * tw[160]
		out[5*32+n1] = (e5 + t5) * tw[5*n1]
		out[21*32+n1] = (e5 - t5) * tw[21*n1]

		t6 := o6 * tw[192]
		out[6*32+n1] = (e6 + t6) * tw[6*n1]
		out[22*32+n1] = (e6 - t6) * tw[22*n1]

		t7 := o7 * tw[224]
		out[7*32+n1] = (e7 + t7) * tw[7*n1]
		out[23*32+n1] = (e7 - t7) * tw[23*n1]

		t8 := o8 * tw[256]
		out[8*32+n1] = (e8 + t8) * tw[8*n1]
		out[24*32+n1] = (e8 - t8) * tw[24*n1]

		t9 := o9 * tw[288]
		out[9*32+n1] = (e9 + t9) * tw[9*n1]
		out[25*32+n1] = (e9 - t9) * tw[25*n1]

		t10 := o10 * tw[320]
		out[10*32+n1] = (e10 + t10) * tw[10*n1]
		out[26*32+n1] = (e10 - t10) * tw[26*n1]

		t11 := o11 * tw[352]
		out[11*32+n1] = (e11 + t11) * tw[11*n1]
		out[27*32+n1] = (e11 - t11) * tw[27*n1]

		t12 := o12 * tw[384]
		out[12*32+n1] = (e12 + t12) * tw[12*n1]
		out[28*32+n1] = (e12 - t12) * tw[28*n1]

		t13 := o13 * tw[416]
		out[13*32+n1] = (e13 + t13) * tw[13*n1]
		out[29*32+n1] = (e13 - t13) * tw[29*n1]

		t14 := o14 * tw[448]
		out[14*32+n1] = (e14 + t14) * tw[14*n1]
		out[30*32+n1] = (e14 - t14) * tw[30*n1]

		t15 := o15 * tw[480]
		out[15*32+n1] = (e15 + t15) * tw[15*n1]
		out[31*32+n1] = (e15 - t15) * tw[31*n1]
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
		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = fft16Complex128(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)

		// FFT-16 on odd elements (bit-reversed input).
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = fft16Complex128(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		// Combine with W_32 twiddle factors and apply inter-stage twiddles W_1024^{k2*n1}.
		out[0*32+n1] = (e0 + o0) * tw[0]
		out[16*32+n1] = (e0 - o0) * tw[16*n1]

		t1 := o1 * tw[32]
		out[1*32+n1] = (e1 + t1) * tw[1*n1]
		out[17*32+n1] = (e1 - t1) * tw[17*n1]

		t2 := o2 * tw[64]
		out[2*32+n1] = (e2 + t2) * tw[2*n1]
		out[18*32+n1] = (e2 - t2) * tw[18*n1]

		t3 := o3 * tw[96]
		out[3*32+n1] = (e3 + t3) * tw[3*n1]
		out[19*32+n1] = (e3 - t3) * tw[19*n1]

		t4 := o4 * tw[128]
		out[4*32+n1] = (e4 + t4) * tw[4*n1]
		out[20*32+n1] = (e4 - t4) * tw[20*n1]

		t5 := o5 * tw[160]
		out[5*32+n1] = (e5 + t5) * tw[5*n1]
		out[21*32+n1] = (e5 - t5) * tw[21*n1]

		t6 := o6 * tw[192]
		out[6*32+n1] = (e6 + t6) * tw[6*n1]
		out[22*32+n1] = (e6 - t6) * tw[22*n1]

		t7 := o7 * tw[224]
		out[7*32+n1] = (e7 + t7) * tw[7*n1]
		out[23*32+n1] = (e7 - t7) * tw[23*n1]

		t8 := o8 * tw[256]
		out[8*32+n1] = (e8 + t8) * tw[8*n1]
		out[24*32+n1] = (e8 - t8) * tw[24*n1]

		t9 := o9 * tw[288]
		out[9*32+n1] = (e9 + t9) * tw[9*n1]
		out[25*32+n1] = (e9 - t9) * tw[25*n1]

		t10 := o10 * tw[320]
		out[10*32+n1] = (e10 + t10) * tw[10*n1]
		out[26*32+n1] = (e10 - t10) * tw[26*n1]

		t11 := o11 * tw[352]
		out[11*32+n1] = (e11 + t11) * tw[11*n1]
		out[27*32+n1] = (e11 - t11) * tw[27*n1]

		t12 := o12 * tw[384]
		out[12*32+n1] = (e12 + t12) * tw[12*n1]
		out[28*32+n1] = (e12 - t12) * tw[28*n1]

		t13 := o13 * tw[416]
		out[13*32+n1] = (e13 + t13) * tw[13*n1]
		out[29*32+n1] = (e13 - t13) * tw[29*n1]

		t14 := o14 * tw[448]
		out[14*32+n1] = (e14 + t14) * tw[14*n1]
		out[30*32+n1] = (e14 - t14) * tw[30*n1]

		t15 := o15 * tw[480]
		out[15*32+n1] = (e15 + t15) * tw[15*n1]
		out[31*32+n1] = (e15 - t15) * tw[31*n1]
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

		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = fft16Complex64(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = fft16Complex64(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		dst[32*0+k2] = e0 + o0
		dst[32*16+k2] = e0 - o0

		t1 := o1 * tw[32]
		dst[32*1+k2] = e1 + t1
		dst[32*17+k2] = e1 - t1

		t2 := o2 * tw[64]
		dst[32*2+k2] = e2 + t2
		dst[32*18+k2] = e2 - t2

		t3 := o3 * tw[96]
		dst[32*3+k2] = e3 + t3
		dst[32*19+k2] = e3 - t3

		t4 := o4 * tw[128]
		dst[32*4+k2] = e4 + t4
		dst[32*20+k2] = e4 - t4

		t5 := o5 * tw[160]
		dst[32*5+k2] = e5 + t5
		dst[32*21+k2] = e5 - t5

		t6 := o6 * tw[192]
		dst[32*6+k2] = e6 + t6
		dst[32*22+k2] = e6 - t6

		t7 := o7 * tw[224]
		dst[32*7+k2] = e7 + t7
		dst[32*23+k2] = e7 - t7

		t8 := o8 * tw[256]
		dst[32*8+k2] = e8 + t8
		dst[32*24+k2] = e8 - t8

		t9 := o9 * tw[288]
		dst[32*9+k2] = e9 + t9
		dst[32*25+k2] = e9 - t9

		t10 := o10 * tw[320]
		dst[32*10+k2] = e10 + t10
		dst[32*26+k2] = e10 - t10

		t11 := o11 * tw[352]
		dst[32*11+k2] = e11 + t11
		dst[32*27+k2] = e11 - t11

		t12 := o12 * tw[384]
		dst[32*12+k2] = e12 + t12
		dst[32*28+k2] = e12 - t12

		t13 := o13 * tw[416]
		dst[32*13+k2] = e13 + t13
		dst[32*29+k2] = e13 - t13

		t14 := o14 * tw[448]
		dst[32*14+k2] = e14 + t14
		dst[32*30+k2] = e14 - t14

		t15 := o15 * tw[480]
		dst[32*15+k2] = e15 + t15
		dst[32*31+k2] = e15 - t15
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
		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 := fft16Complex64Inverse(
			s[32*0+k2], s[32*16+k2], s[32*8+k2], s[32*24+k2], s[32*4+k2], s[32*20+k2], s[32*12+k2], s[32*28+k2],
			s[32*2+k2], s[32*18+k2], s[32*10+k2], s[32*26+k2], s[32*6+k2], s[32*22+k2], s[32*14+k2], s[32*30+k2])

		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 := fft16Complex64Inverse(
			s[32*1+k2], s[32*17+k2], s[32*9+k2], s[32*25+k2], s[32*5+k2], s[32*21+k2], s[32*13+k2], s[32*29+k2],
			s[32*3+k2], s[32*19+k2], s[32*11+k2], s[32*27+k2], s[32*7+k2], s[32*23+k2], s[32*15+k2], s[32*31+k2])

		r0 := e0 + o0
		r16 := e0 - o0

		t1 := o1 * complex(real(tw[32]), -imag(tw[32]))
		r1 := e1 + t1
		r17 := e1 - t1

		t2 := o2 * complex(real(tw[64]), -imag(tw[64]))
		r2 := e2 + t2
		r18 := e2 - t2

		t3 := o3 * complex(real(tw[96]), -imag(tw[96]))
		r3 := e3 + t3
		r19 := e3 - t3

		t4 := o4 * complex(real(tw[128]), -imag(tw[128]))
		r4 := e4 + t4
		r20 := e4 - t4

		t5 := o5 * complex(real(tw[160]), -imag(tw[160]))
		r5 := e5 + t5
		r21 := e5 - t5

		t6 := o6 * complex(real(tw[192]), -imag(tw[192]))
		r6 := e6 + t6
		r22 := e6 - t6

		t7 := o7 * complex(real(tw[224]), -imag(tw[224]))
		r7 := e7 + t7
		r23 := e7 - t7

		t8 := o8 * complex(real(tw[256]), -imag(tw[256]))
		r8 := e8 + t8
		r24 := e8 - t8

		t9 := o9 * complex(real(tw[288]), -imag(tw[288]))
		r9 := e9 + t9
		r25 := e9 - t9

		t10 := o10 * complex(real(tw[320]), -imag(tw[320]))
		r10 := e10 + t10
		r26 := e10 - t10

		t11 := o11 * complex(real(tw[352]), -imag(tw[352]))
		r11 := e11 + t11
		r27 := e11 - t11

		t12 := o12 * complex(real(tw[384]), -imag(tw[384]))
		r12 := e12 + t12
		r28 := e12 - t12

		t13 := o13 * complex(real(tw[416]), -imag(tw[416]))
		r13 := e13 + t13
		r29 := e13 - t13

		t14 := o14 * complex(real(tw[448]), -imag(tw[448]))
		r14 := e14 + t14
		r30 := e14 - t14

		t15 := o15 * complex(real(tw[480]), -imag(tw[480]))
		r15 := e15 + t15
		r31 := e15 - t15

		base := k2 * 32
		out[base+0] = r0 * complex(real(tw[k2*0]), -imag(tw[k2*0]))
		out[base+1] = r1 * complex(real(tw[k2*1]), -imag(tw[k2*1]))
		out[base+2] = r2 * complex(real(tw[k2*2]), -imag(tw[k2*2]))
		out[base+3] = r3 * complex(real(tw[k2*3]), -imag(tw[k2*3]))
		out[base+4] = r4 * complex(real(tw[k2*4]), -imag(tw[k2*4]))
		out[base+5] = r5 * complex(real(tw[k2*5]), -imag(tw[k2*5]))
		out[base+6] = r6 * complex(real(tw[k2*6]), -imag(tw[k2*6]))
		out[base+7] = r7 * complex(real(tw[k2*7]), -imag(tw[k2*7]))
		out[base+8] = r8 * complex(real(tw[k2*8]), -imag(tw[k2*8]))
		out[base+9] = r9 * complex(real(tw[k2*9]), -imag(tw[k2*9]))
		out[base+10] = r10 * complex(real(tw[k2*10]), -imag(tw[k2*10]))
		out[base+11] = r11 * complex(real(tw[k2*11]), -imag(tw[k2*11]))
		out[base+12] = r12 * complex(real(tw[k2*12]), -imag(tw[k2*12]))
		out[base+13] = r13 * complex(real(tw[k2*13]), -imag(tw[k2*13]))
		out[base+14] = r14 * complex(real(tw[k2*14]), -imag(tw[k2*14]))
		out[base+15] = r15 * complex(real(tw[k2*15]), -imag(tw[k2*15]))
		out[base+16] = r16 * complex(real(tw[k2*16]), -imag(tw[k2*16]))
		out[base+17] = r17 * complex(real(tw[k2*17]), -imag(tw[k2*17]))
		out[base+18] = r18 * complex(real(tw[k2*18]), -imag(tw[k2*18]))
		out[base+19] = r19 * complex(real(tw[k2*19]), -imag(tw[k2*19]))
		out[base+20] = r20 * complex(real(tw[k2*20]), -imag(tw[k2*20]))
		out[base+21] = r21 * complex(real(tw[k2*21]), -imag(tw[k2*21]))
		out[base+22] = r22 * complex(real(tw[k2*22]), -imag(tw[k2*22]))
		out[base+23] = r23 * complex(real(tw[k2*23]), -imag(tw[k2*23]))
		out[base+24] = r24 * complex(real(tw[k2*24]), -imag(tw[k2*24]))
		out[base+25] = r25 * complex(real(tw[k2*25]), -imag(tw[k2*25]))
		out[base+26] = r26 * complex(real(tw[k2*26]), -imag(tw[k2*26]))
		out[base+27] = r27 * complex(real(tw[k2*27]), -imag(tw[k2*27]))
		out[base+28] = r28 * complex(real(tw[k2*28]), -imag(tw[k2*28]))
		out[base+29] = r29 * complex(real(tw[k2*29]), -imag(tw[k2*29]))
		out[base+30] = r30 * complex(real(tw[k2*30]), -imag(tw[k2*30]))
		out[base+31] = r31 * complex(real(tw[k2*31]), -imag(tw[k2*31]))
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

		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = fft16Complex64Inverse(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = fft16Complex64Inverse(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		scaleComplex := complex(scale, 0)
		dst[32*0+n1] = (e0 + o0) * scaleComplex
		dst[32*16+n1] = (e0 - o0) * scaleComplex

		t1 := o1 * complex(real(tw[32]), -imag(tw[32]))
		dst[32*1+n1] = (e1 + t1) * scaleComplex
		dst[32*17+n1] = (e1 - t1) * scaleComplex

		t2 := o2 * complex(real(tw[64]), -imag(tw[64]))
		dst[32*2+n1] = (e2 + t2) * scaleComplex
		dst[32*18+n1] = (e2 - t2) * scaleComplex

		t3 := o3 * complex(real(tw[96]), -imag(tw[96]))
		dst[32*3+n1] = (e3 + t3) * scaleComplex
		dst[32*19+n1] = (e3 - t3) * scaleComplex

		t4 := o4 * complex(real(tw[128]), -imag(tw[128]))
		dst[32*4+n1] = (e4 + t4) * scaleComplex
		dst[32*20+n1] = (e4 - t4) * scaleComplex

		t5 := o5 * complex(real(tw[160]), -imag(tw[160]))
		dst[32*5+n1] = (e5 + t5) * scaleComplex
		dst[32*21+n1] = (e5 - t5) * scaleComplex

		t6 := o6 * complex(real(tw[192]), -imag(tw[192]))
		dst[32*6+n1] = (e6 + t6) * scaleComplex
		dst[32*22+n1] = (e6 - t6) * scaleComplex

		t7 := o7 * complex(real(tw[224]), -imag(tw[224]))
		dst[32*7+n1] = (e7 + t7) * scaleComplex
		dst[32*23+n1] = (e7 - t7) * scaleComplex

		t8 := o8 * complex(real(tw[256]), -imag(tw[256]))
		dst[32*8+n1] = (e8 + t8) * scaleComplex
		dst[32*24+n1] = (e8 - t8) * scaleComplex

		t9 := o9 * complex(real(tw[288]), -imag(tw[288]))
		dst[32*9+n1] = (e9 + t9) * scaleComplex
		dst[32*25+n1] = (e9 - t9) * scaleComplex

		t10 := o10 * complex(real(tw[320]), -imag(tw[320]))
		dst[32*10+n1] = (e10 + t10) * scaleComplex
		dst[32*26+n1] = (e10 - t10) * scaleComplex

		t11 := o11 * complex(real(tw[352]), -imag(tw[352]))
		dst[32*11+n1] = (e11 + t11) * scaleComplex
		dst[32*27+n1] = (e11 - t11) * scaleComplex

		t12 := o12 * complex(real(tw[384]), -imag(tw[384]))
		dst[32*12+n1] = (e12 + t12) * scaleComplex
		dst[32*28+n1] = (e12 - t12) * scaleComplex

		t13 := o13 * complex(real(tw[416]), -imag(tw[416]))
		dst[32*13+n1] = (e13 + t13) * scaleComplex
		dst[32*29+n1] = (e13 - t13) * scaleComplex

		t14 := o14 * complex(real(tw[448]), -imag(tw[448]))
		dst[32*14+n1] = (e14 + t14) * scaleComplex
		dst[32*30+n1] = (e14 - t14) * scaleComplex

		t15 := o15 * complex(real(tw[480]), -imag(tw[480]))
		dst[32*15+n1] = (e15 + t15) * scaleComplex
		dst[32*31+n1] = (e15 - t15) * scaleComplex
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

		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = fft16Complex128(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = fft16Complex128(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		dst[32*0+k2] = e0 + o0
		dst[32*16+k2] = e0 - o0

		t1 := o1 * tw[32]
		dst[32*1+k2] = e1 + t1
		dst[32*17+k2] = e1 - t1

		t2 := o2 * tw[64]
		dst[32*2+k2] = e2 + t2
		dst[32*18+k2] = e2 - t2

		t3 := o3 * tw[96]
		dst[32*3+k2] = e3 + t3
		dst[32*19+k2] = e3 - t3

		t4 := o4 * tw[128]
		dst[32*4+k2] = e4 + t4
		dst[32*20+k2] = e4 - t4

		t5 := o5 * tw[160]
		dst[32*5+k2] = e5 + t5
		dst[32*21+k2] = e5 - t5

		t6 := o6 * tw[192]
		dst[32*6+k2] = e6 + t6
		dst[32*22+k2] = e6 - t6

		t7 := o7 * tw[224]
		dst[32*7+k2] = e7 + t7
		dst[32*23+k2] = e7 - t7

		t8 := o8 * tw[256]
		dst[32*8+k2] = e8 + t8
		dst[32*24+k2] = e8 - t8

		t9 := o9 * tw[288]
		dst[32*9+k2] = e9 + t9
		dst[32*25+k2] = e9 - t9

		t10 := o10 * tw[320]
		dst[32*10+k2] = e10 + t10
		dst[32*26+k2] = e10 - t10

		t11 := o11 * tw[352]
		dst[32*11+k2] = e11 + t11
		dst[32*27+k2] = e11 - t11

		t12 := o12 * tw[384]
		dst[32*12+k2] = e12 + t12
		dst[32*28+k2] = e12 - t12

		t13 := o13 * tw[416]
		dst[32*13+k2] = e13 + t13
		dst[32*29+k2] = e13 - t13

		t14 := o14 * tw[448]
		dst[32*14+k2] = e14 + t14
		dst[32*30+k2] = e14 - t14

		t15 := o15 * tw[480]
		dst[32*15+k2] = e15 + t15
		dst[32*31+k2] = e15 - t15
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
		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 := fft16Complex128Inverse(
			s[32*0+k2], s[32*16+k2], s[32*8+k2], s[32*24+k2], s[32*4+k2], s[32*20+k2], s[32*12+k2], s[32*28+k2],
			s[32*2+k2], s[32*18+k2], s[32*10+k2], s[32*26+k2], s[32*6+k2], s[32*22+k2], s[32*14+k2], s[32*30+k2])
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 := fft16Complex128Inverse(
			s[32*1+k2], s[32*17+k2], s[32*9+k2], s[32*25+k2], s[32*5+k2], s[32*21+k2], s[32*13+k2], s[32*29+k2],
			s[32*3+k2], s[32*19+k2], s[32*11+k2], s[32*27+k2], s[32*7+k2], s[32*23+k2], s[32*15+k2], s[32*31+k2])

		r0 := e0 + o0
		r16 := e0 - o0

		t1 := o1 * complex(real(tw[32]), -imag(tw[32]))
		r1 := e1 + t1
		r17 := e1 - t1

		t2 := o2 * complex(real(tw[64]), -imag(tw[64]))
		r2 := e2 + t2
		r18 := e2 - t2

		t3 := o3 * complex(real(tw[96]), -imag(tw[96]))
		r3 := e3 + t3
		r19 := e3 - t3

		t4 := o4 * complex(real(tw[128]), -imag(tw[128]))
		r4 := e4 + t4
		r20 := e4 - t4

		t5 := o5 * complex(real(tw[160]), -imag(tw[160]))
		r5 := e5 + t5
		r21 := e5 - t5

		t6 := o6 * complex(real(tw[192]), -imag(tw[192]))
		r6 := e6 + t6
		r22 := e6 - t6

		t7 := o7 * complex(real(tw[224]), -imag(tw[224]))
		r7 := e7 + t7
		r23 := e7 - t7

		t8 := o8 * complex(real(tw[256]), -imag(tw[256]))
		r8 := e8 + t8
		r24 := e8 - t8

		t9 := o9 * complex(real(tw[288]), -imag(tw[288]))
		r9 := e9 + t9
		r25 := e9 - t9

		t10 := o10 * complex(real(tw[320]), -imag(tw[320]))
		r10 := e10 + t10
		r26 := e10 - t10

		t11 := o11 * complex(real(tw[352]), -imag(tw[352]))
		r11 := e11 + t11
		r27 := e11 - t11

		t12 := o12 * complex(real(tw[384]), -imag(tw[384]))
		r12 := e12 + t12
		r28 := e12 - t12

		t13 := o13 * complex(real(tw[416]), -imag(tw[416]))
		r13 := e13 + t13
		r29 := e13 - t13

		t14 := o14 * complex(real(tw[448]), -imag(tw[448]))
		r14 := e14 + t14
		r30 := e14 - t14

		t15 := o15 * complex(real(tw[480]), -imag(tw[480]))
		r15 := e15 + t15
		r31 := e15 - t15

		base := k2 * 32
		out[base+0] = r0 * complex(real(tw[k2*0]), -imag(tw[k2*0]))
		out[base+1] = r1 * complex(real(tw[k2*1]), -imag(tw[k2*1]))
		out[base+2] = r2 * complex(real(tw[k2*2]), -imag(tw[k2*2]))
		out[base+3] = r3 * complex(real(tw[k2*3]), -imag(tw[k2*3]))
		out[base+4] = r4 * complex(real(tw[k2*4]), -imag(tw[k2*4]))
		out[base+5] = r5 * complex(real(tw[k2*5]), -imag(tw[k2*5]))
		out[base+6] = r6 * complex(real(tw[k2*6]), -imag(tw[k2*6]))
		out[base+7] = r7 * complex(real(tw[k2*7]), -imag(tw[k2*7]))
		out[base+8] = r8 * complex(real(tw[k2*8]), -imag(tw[k2*8]))
		out[base+9] = r9 * complex(real(tw[k2*9]), -imag(tw[k2*9]))
		out[base+10] = r10 * complex(real(tw[k2*10]), -imag(tw[k2*10]))
		out[base+11] = r11 * complex(real(tw[k2*11]), -imag(tw[k2*11]))
		out[base+12] = r12 * complex(real(tw[k2*12]), -imag(tw[k2*12]))
		out[base+13] = r13 * complex(real(tw[k2*13]), -imag(tw[k2*13]))
		out[base+14] = r14 * complex(real(tw[k2*14]), -imag(tw[k2*14]))
		out[base+15] = r15 * complex(real(tw[k2*15]), -imag(tw[k2*15]))
		out[base+16] = r16 * complex(real(tw[k2*16]), -imag(tw[k2*16]))
		out[base+17] = r17 * complex(real(tw[k2*17]), -imag(tw[k2*17]))
		out[base+18] = r18 * complex(real(tw[k2*18]), -imag(tw[k2*18]))
		out[base+19] = r19 * complex(real(tw[k2*19]), -imag(tw[k2*19]))
		out[base+20] = r20 * complex(real(tw[k2*20]), -imag(tw[k2*20]))
		out[base+21] = r21 * complex(real(tw[k2*21]), -imag(tw[k2*21]))
		out[base+22] = r22 * complex(real(tw[k2*22]), -imag(tw[k2*22]))
		out[base+23] = r23 * complex(real(tw[k2*23]), -imag(tw[k2*23]))
		out[base+24] = r24 * complex(real(tw[k2*24]), -imag(tw[k2*24]))
		out[base+25] = r25 * complex(real(tw[k2*25]), -imag(tw[k2*25]))
		out[base+26] = r26 * complex(real(tw[k2*26]), -imag(tw[k2*26]))
		out[base+27] = r27 * complex(real(tw[k2*27]), -imag(tw[k2*27]))
		out[base+28] = r28 * complex(real(tw[k2*28]), -imag(tw[k2*28]))
		out[base+29] = r29 * complex(real(tw[k2*29]), -imag(tw[k2*29]))
		out[base+30] = r30 * complex(real(tw[k2*30]), -imag(tw[k2*30]))
		out[base+31] = r31 * complex(real(tw[k2*31]), -imag(tw[k2*31]))
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

		e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = fft16Complex128Inverse(
			e0, e8, e4, e12, e2, e10, e6, e14, e1, e9, e5, e13, e3, e11, e7, e15)
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = fft16Complex128Inverse(
			o0, o8, o4, o12, o2, o10, o6, o14, o1, o9, o5, o13, o3, o11, o7, o15)

		dst[32*0+n1] = (e0 + o0) * scale
		dst[32*16+n1] = (e0 - o0) * scale

		t1 := o1 * complex(real(tw[32]), -imag(tw[32]))
		dst[32*1+n1] = (e1 + t1) * scale
		dst[32*17+n1] = (e1 - t1) * scale

		t2 := o2 * complex(real(tw[64]), -imag(tw[64]))
		dst[32*2+n1] = (e2 + t2) * scale
		dst[32*18+n1] = (e2 - t2) * scale

		t3 := o3 * complex(real(tw[96]), -imag(tw[96]))
		dst[32*3+n1] = (e3 + t3) * scale
		dst[32*19+n1] = (e3 - t3) * scale

		t4 := o4 * complex(real(tw[128]), -imag(tw[128]))
		dst[32*4+n1] = (e4 + t4) * scale
		dst[32*20+n1] = (e4 - t4) * scale

		t5 := o5 * complex(real(tw[160]), -imag(tw[160]))
		dst[32*5+n1] = (e5 + t5) * scale
		dst[32*21+n1] = (e5 - t5) * scale

		t6 := o6 * complex(real(tw[192]), -imag(tw[192]))
		dst[32*6+n1] = (e6 + t6) * scale
		dst[32*22+n1] = (e6 - t6) * scale

		t7 := o7 * complex(real(tw[224]), -imag(tw[224]))
		dst[32*7+n1] = (e7 + t7) * scale
		dst[32*23+n1] = (e7 - t7) * scale

		t8 := o8 * complex(real(tw[256]), -imag(tw[256]))
		dst[32*8+n1] = (e8 + t8) * scale
		dst[32*24+n1] = (e8 - t8) * scale

		t9 := o9 * complex(real(tw[288]), -imag(tw[288]))
		dst[32*9+n1] = (e9 + t9) * scale
		dst[32*25+n1] = (e9 - t9) * scale

		t10 := o10 * complex(real(tw[320]), -imag(tw[320]))
		dst[32*10+n1] = (e10 + t10) * scale
		dst[32*26+n1] = (e10 - t10) * scale

		t11 := o11 * complex(real(tw[352]), -imag(tw[352]))
		dst[32*11+n1] = (e11 + t11) * scale
		dst[32*27+n1] = (e11 - t11) * scale

		t12 := o12 * complex(real(tw[384]), -imag(tw[384]))
		dst[32*12+n1] = (e12 + t12) * scale
		dst[32*28+n1] = (e12 - t12) * scale

		t13 := o13 * complex(real(tw[416]), -imag(tw[416]))
		dst[32*13+n1] = (e13 + t13) * scale
		dst[32*29+n1] = (e13 - t13) * scale

		t14 := o14 * complex(real(tw[448]), -imag(tw[448]))
		dst[32*14+n1] = (e14 + t14) * scale
		dst[32*30+n1] = (e14 - t14) * scale

		t15 := o15 * complex(real(tw[480]), -imag(tw[480]))
		dst[32*15+n1] = (e15 + t15) * scale
		dst[32*31+n1] = (e15 - t15) * scale
	}

	return true
}
