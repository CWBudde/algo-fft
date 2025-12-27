package fft

func forwardDIT256Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*64]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for base := 0; base < n; base += 128 {
		for j := range 64 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+64], tw)
			work[base+j] = a
			work[base+j+64] = b
		}
	}

	for j := range 128 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+128], tw)
		work[j] = a
		work[j+128] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT256Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*64]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for base := 0; base < n; base += 128 {
		for j := range 64 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+64], tw)
			work[base+j] = a
			work[base+j+64] = b
		}
	}

	for j := range 128 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+128], tw)
		work[j] = a
		work[j+128] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

func forwardDIT256Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*64]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for base := 0; base < n; base += 128 {
		for j := range 64 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+64], tw)
			work[base+j] = a
			work[base+j+64] = b
		}
	}

	for j := range 128 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+128], tw)
		work[j] = a
		work[j+128] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

func inverseDIT256Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*64]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	for base := 0; base < n; base += 128 {
		for j := range 64 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+64], tw)
			work[base+j] = a
			work[base+j+64] = b
		}
	}

	for j := range 128 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+128], tw)
		work[j] = a
		work[j+128] = b
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
