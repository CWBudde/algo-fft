package kernels

func radix5TransformComplex64(dst, src, twiddle, scratch []complex64, bitrev []int, inverse bool) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	if !isPowerOf5(n) {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	digits := logBase5(n)
	for i := range n {
		work[i] = src[reverseBase5(i, digits)]
	}

	useAVX2 := radix5AVX2Available()

	for size := 5; size <= n; size *= 5 {
		span := size / 5
		step := n / size

		for base := 0; base < n; base += size {
			j := 0
			if useAVX2 && span >= 2 {
				for ; j+1 < span; j += 2 {
					idx0 := base + j
					idx1 := idx0 + span
					idx2 := idx1 + span
					idx3 := idx2 + span
					idx4 := idx3 + span

					idx0b := idx0 + 1
					idx1b := idx1 + 1
					idx2b := idx2 + 1
					idx3b := idx3 + 1
					idx4b := idx4 + 1

					w1a := twiddle[j*step]
					w2a := twiddle[2*j*step]
					w3a := twiddle[3*j*step]
					w4a := twiddle[4*j*step]
					w1b := twiddle[(j+1)*step]
					w2b := twiddle[2*(j+1)*step]
					w3b := twiddle[3*(j+1)*step]
					w4b := twiddle[4*(j+1)*step]

					if inverse {
						w1a = conj(w1a)
						w2a = conj(w2a)
						w3a = conj(w3a)
						w4a = conj(w4a)
						w1b = conj(w1b)
						w2b = conj(w2b)
						w3b = conj(w3b)
						w4b = conj(w4b)
					}

					var (
						a0 [2]complex64
						a1 [2]complex64
						a2 [2]complex64
						a3 [2]complex64
						a4 [2]complex64
					)

					a0[0] = work[idx0]
					a0[1] = work[idx0b]
					a1[0] = w1a * work[idx1]
					a1[1] = w1b * work[idx1b]
					a2[0] = w2a * work[idx2]
					a2[1] = w2b * work[idx2b]
					a3[0] = w3a * work[idx3]
					a3[1] = w3b * work[idx3b]
					a4[0] = w4a * work[idx4]
					a4[1] = w4b * work[idx4b]

					var (
						y0 [2]complex64
						y1 [2]complex64
						y2 [2]complex64
						y3 [2]complex64
						y4 [2]complex64
					)

					if inverse {
						butterfly5InverseAVX2Complex64Slices(
							y0[:], y1[:], y2[:], y3[:], y4[:],
							a0[:], a1[:], a2[:], a3[:], a4[:],
						)
					} else {
						butterfly5ForwardAVX2Complex64Slices(
							y0[:], y1[:], y2[:], y3[:], y4[:],
							a0[:], a1[:], a2[:], a3[:], a4[:],
						)
					}

					work[idx0] = y0[0]
					work[idx0b] = y0[1]
					work[idx1] = y1[0]
					work[idx1b] = y1[1]
					work[idx2] = y2[0]
					work[idx2b] = y2[1]
					work[idx3] = y3[0]
					work[idx3b] = y3[1]
					work[idx4] = y4[0]
					work[idx4b] = y4[1]
				}
			}

			for ; j < span; j++ {
				idx0 := base + j
				idx1 := idx0 + span
				idx2 := idx1 + span
				idx3 := idx2 + span
				idx4 := idx3 + span

				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]
				w3 := twiddle[3*j*step]
				w4 := twiddle[4*j*step]

				if inverse {
					w1 = conj(w1)
					w2 = conj(w2)
					w3 = conj(w3)
					w4 = conj(w4)
				}

				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]
				a3 := w3 * work[idx3]
				a4 := w4 * work[idx4]

				var y0, y1, y2, y3, y4 complex64
				if inverse {
					y0, y1, y2, y3, y4 = butterfly5InverseComplex64(a0, a1, a2, a3, a4)
				} else {
					y0, y1, y2, y3, y4 = butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
				}

				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
				work[idx3] = y3
				work[idx4] = y4
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	if inverse {
		scale := complex(float32(1.0/float32(n)), 0)
		for i := range dst {
			dst[i] *= scale
		}
	}

	return true
}
