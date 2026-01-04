package fft

import "github.com/MeKo-Christian/algo-fft/internal/kernels"

const mixedRadixMaxStages = 64

func forwardMixedRadixComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return mixedRadixForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseMixedRadixComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return mixedRadixInverse[complex64](dst, src, twiddle, scratch, bitrev)
}

func forwardMixedRadixComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return mixedRadixForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseMixedRadixComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return mixedRadixInverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func mixedRadixForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, bitrev, false)
}

func mixedRadixInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, bitrev, true)
}

func mixedRadixTransform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
	_ = bitrev

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

	var radices [mixedRadixMaxStages]int

	stageCount := mixedRadixSchedule(n, &radices)
	if stageCount == 0 {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Ping-pong buffering: recursively alternate between buffers to eliminate
	// intermediate copies. Only copy at the end if result isn't already in dst.
	// Use type-specific implementation to avoid generic overhead
	var zero T
	switch any(zero).(type) {
	case complex64:
		mixedRadixRecursivePingPongComplex64(
			any(work).([]complex64),
			any(src).([]complex64),
			any(scratch).([]complex64),
			n, 1, 1, radices[:stageCount],
			any(twiddle).([]complex64),
			inverse,
		)
	case complex128:
		mixedRadixRecursivePingPongComplex128(
			any(work).([]complex128),
			any(src).([]complex128),
			any(scratch).([]complex128),
			n, 1, 1, radices[:stageCount],
			any(twiddle).([]complex128),
			inverse,
		)
	default:
		return false
	}

	if !workIsDst {
		copy(dst, work)
	}

	if inverse {
		scale := complexFromFloat64[T](1.0/float64(n), 0)
		for i := range dst {
			dst[i] *= scale
		}
	}

	return true
}

func mixedRadixSchedule(n int, radices *[mixedRadixMaxStages]int) int {
	if n < 2 {
		return 0
	}

	count := 0

	for n > 1 {
		switch {
		case n%5 == 0:
			radices[count] = 5
			n /= 5
		case n%4 == 0:
			radices[count] = 4
			n /= 4
		case n%3 == 0:
			radices[count] = 3
			n /= 3
		case n%2 == 0:
			radices[count] = 2
			n /= 2
		default:
			return 0
		}

		count++
		if count >= mixedRadixMaxStages {
			return 0
		}
	}

	return count
}

// mixedRadixRecursivePingPongComplex64 is a specialized complex64 version that calls
// type-specific butterfly functions to avoid generic overhead.
func mixedRadixRecursivePingPongComplex64(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool) {
	if n == 1 {
		dst[0] = src[0]
		return
	}

	radix := radices[0]
	span := n / radix
	nextRadices := radices[1:]

	// Recursively process sub-transforms
	for j := range radix {
		if len(nextRadices) == 0 {
			dst[j*span] = src[j*stride]
		} else {
			mixedRadixRecursivePingPongComplex64(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse)
		}
	}

	// Determine where the recursive calls wrote their data
	var input []complex64
	if len(nextRadices) == 0 {
		input = dst
	} else {
		input = work
	}

	// Apply radix-r butterfly with type-specific functions
	for k := range span {
		switch radix {
		case 2:
			w1 := twiddle[k*step]
			if inverse {
				w1 = conj(w1)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]

			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		case 3:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]

			var y0, y1, y2 complex64
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex64(a0, a1, a2)
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex64(a0, a1, a2)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
		case 4:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]
			a3 := w3 * input[3*span+k]

			var y0, y1, y2, y3 complex64
			if inverse {
				y0, y1, y2, y3 = kernels.Butterfly4InverseComplex64(a0, a1, a2, a3)
			} else {
				y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex64(a0, a1, a2, a3)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
		case 5:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]
			a3 := w3 * input[3*span+k]
			a4 := w4 * input[4*span+k]

			var y0, y1, y2, y3, y4 complex64
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex64(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
		default:
			return
		}
	}
}

// mixedRadixRecursivePingPongComplex128 is a specialized complex128 version that calls
// type-specific butterfly functions to avoid generic overhead.
func mixedRadixRecursivePingPongComplex128(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool) {
	if n == 1 {
		dst[0] = src[0]
		return
	}

	radix := radices[0]
	span := n / radix
	nextRadices := radices[1:]

	// Recursively process sub-transforms
	for j := range radix {
		if len(nextRadices) == 0 {
			dst[j*span] = src[j*stride]
		} else {
			mixedRadixRecursivePingPongComplex128(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse)
		}
	}

	// Determine where the recursive calls wrote their data
	var input []complex128
	if len(nextRadices) == 0 {
		input = dst
	} else {
		input = work
	}

	// Apply radix-r butterfly with type-specific functions
	for k := range span {
		switch radix {
		case 2:
			w1 := twiddle[k*step]
			if inverse {
				w1 = conj(w1)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]

			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		case 3:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]

			var y0, y1, y2 complex128
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex128(a0, a1, a2)
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex128(a0, a1, a2)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
		case 4:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]
			a3 := w3 * input[3*span+k]

			var y0, y1, y2, y3 complex128
			if inverse {
				y0, y1, y2, y3 = kernels.Butterfly4InverseComplex128(a0, a1, a2, a3)
			} else {
				y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex128(a0, a1, a2, a3)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
		case 5:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]
			a3 := w3 * input[3*span+k]
			a4 := w4 * input[4*span+k]

			var y0, y1, y2, y3, y4 complex128
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex128(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex128(a0, a1, a2, a3, a4)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
		default:
			return
		}
	}
}

// mixedRadixRecursivePingPong implements mixed-radix FFT with ping-pong buffering
// to eliminate intermediate memory copies. Buffers alternate between dst and work
// at each recursive level.
//
// Key optimization: Instead of copying results after each butterfly operation,
// we alternate which buffer we write to at each recursion level. This eliminates
// the costly copy() operation that was consuming 14% of execution time.
//
// Parameters:
//   - dst: output buffer for this stage (where final result should be)
//   - src: input buffer for this stage
//   - work: alternate working buffer (swapped with dst at recursive calls)
func mixedRadixRecursivePingPong[T Complex](dst, src, work []T, n, stride, step int, radices []int, twiddle []T, inverse bool) {
	if n == 1 {
		dst[0] = src[0]
		return
	}

	radix := radices[0]
	span := n / radix
	nextRadices := radices[1:]

	// Recursively process sub-transforms
	// Key: we swap dst and work for recursive calls to ping-pong between buffers
	for j := range radix {
		if len(nextRadices) == 0 {
			// Base case: no more stages, just copy data
			dst[j*span] = src[j*stride]
		} else {
			// Recursive case: swap buffers (write to work, use dst as scratch)
			mixedRadixRecursivePingPong(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse)
		}
	}

	// Determine where the recursive calls wrote their data
	var input []T
	if len(nextRadices) == 0 {
		// Base case: data is in dst (we just copied it above)
		input = dst
	} else {
		// Recursive case: data is in work (recursive calls wrote there)
		input = work
	}

	// Apply radix-r butterfly, reading from input and writing to dst
	for k := range span {
		switch radix {
		case 2:
			w1 := twiddle[k*step]
			if inverse {
				w1 = conj(w1)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]

			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		case 3:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]

			var y0, y1, y2 T
			if inverse {
				y0, y1, y2 = butterfly3Inverse(a0, a1, a2)
			} else {
				y0, y1, y2 = butterfly3Forward(a0, a1, a2)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
		case 4:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]
			a3 := w3 * input[3*span+k]

			var y0, y1, y2, y3 T
			if inverse {
				y0, y1, y2, y3 = butterfly4Inverse(a0, a1, a2, a3)
			} else {
				y0, y1, y2, y3 = butterfly4Forward(a0, a1, a2, a3)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
		case 5:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
			}

			a0 := input[k]
			a1 := w1 * input[span+k]
			a2 := w2 * input[2*span+k]
			a3 := w3 * input[3*span+k]
			a4 := w4 * input[4*span+k]

			var y0, y1, y2, y3, y4 T
			if inverse {
				y0, y1, y2, y3, y4 = butterfly5Inverse(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = butterfly5Forward(a0, a1, a2, a3, a4)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
		default:
			return
		}
	}
}
