package kernels

import (
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// radix4TransformComplex64 is the monomorphized complex64 twin of
// radix4Transform, matching the radix3TransformComplex64 /
// radix5TransformComplex64 pattern.
//
// It exists for two reasons the generic version cannot serve: the stage
// twiddle products go through mathpkg.MulComplex64 and so stay in single
// precision (the generic `w * work[idx]` promotes to float64 — see
// mathpkg.MulComplex64), and the butterfly is called directly instead of
// through the any()-typeswitch dispatcher in butterfly4Forward.
func radix4TransformComplex64(dst, src, twiddle, scratch []complex64, inverse bool) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	if !isPowerOf4(n) {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	digits := log2(n) / 2
	for i := range n {
		work[i] = src[reverseBase4(i, digits)]
	}

	for size := 4; size <= n; size *= 4 {
		quarter := size / 4

		step := n / size
		for base := 0; base < n; base += size {
			for j := range quarter {
				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]
				w3 := twiddle[3*j*step]

				if inverse {
					w1 = conj(w1)
					w2 = conj(w2)
					w3 = conj(w3)
				}

				a0 := work[idx0]
				a1 := mathpkg.MulComplex64(w1, work[idx1])
				a2 := mathpkg.MulComplex64(w2, work[idx2])
				a3 := mathpkg.MulComplex64(w3, work[idx3])

				var y0, y1, y2, y3 complex64
				if inverse {
					y0, y1, y2, y3 = butterfly4InverseComplex64(a0, a1, a2, a3)
				} else {
					y0, y1, y2, y3 = butterfly4ForwardComplex64(a0, a1, a2, a3)
				}

				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
				work[idx3] = y3
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	if inverse {
		scale := float32(1.0 / float64(n))
		for i := range dst {
			dst[i] = complex(real(dst[i])*scale, imag(dst[i])*scale)
		}
	}

	return true
}
