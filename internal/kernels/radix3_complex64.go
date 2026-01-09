package kernels

func radix3TransformComplex64(dst, src, twiddle, scratch []complex64, bitrev []int, inverse bool) bool {
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

	if !isPowerOf3(n) {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	digits := logBase3(n)
	for i := range n {
		work[i] = src[reverseBase3(i, digits)]
	}

	useAVX2 := radix3AVX2Available()

	for size := 3; size <= n; size *= 3 {
		third := size / 3
		step := n / size

		for base := 0; base < n; base += size {
			j := 0
			if useAVX2 && third >= 4 {
				for ; j+3 < third; j += 4 {
					idx0 := base + j
					idx1 := idx0 + third
					idx2 := idx1 + third

					var (
						a0 [4]complex64
						a1 [4]complex64
						a2 [4]complex64
					)

					for lane := range 4 {
						jj := j + lane
						w1 := twiddle[jj*step]
						w2 := twiddle[2*jj*step]

						if inverse {
							w1 = conj(w1)
							w2 = conj(w2)
						}

						a0[lane] = work[idx0+lane]
						a1[lane] = w1 * work[idx1+lane]
						a2[lane] = w2 * work[idx2+lane]
					}

					var (
						y0 [4]complex64
						y1 [4]complex64
						y2 [4]complex64
					)

					if inverse {
						butterfly3InverseAVX2Complex64Slices(y0[:], y1[:], y2[:], a0[:], a1[:], a2[:])
					} else {
						butterfly3ForwardAVX2Complex64Slices(y0[:], y1[:], y2[:], a0[:], a1[:], a2[:])
					}

					for lane := range 4 {
						work[idx0+lane] = y0[lane]
						work[idx1+lane] = y1[lane]
						work[idx2+lane] = y2[lane]
					}
				}
			}

			for ; j < third; j++ {
				idx0 := base + j
				idx1 := idx0 + third
				idx2 := idx1 + third

				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]

				if inverse {
					w1 = conj(w1)
					w2 = conj(w2)
				}

				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]

				var y0, y1, y2 complex64
				if inverse {
					y0, y1, y2 = butterfly3InverseComplex64(a0, a1, a2)
				} else {
					y0, y1, y2 = butterfly3ForwardComplex64(a0, a1, a2)
				}

				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
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
