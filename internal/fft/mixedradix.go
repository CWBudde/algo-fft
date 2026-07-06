package fft

import (
	"strconv"

	"github.com/cwbudde/algo-fft/internal/kernels"
)

const mixedRadixMaxStages = 64

func forwardMixedRadixComplex64(dst, src, twiddle, scratch []complex64) bool {
	return mixedRadixForward[complex64](dst, src, twiddle, scratch)
}

func inverseMixedRadixComplex64(dst, src, twiddle, scratch []complex64) bool {
	return mixedRadixInverse[complex64](dst, src, twiddle, scratch)
}

func forwardMixedRadixComplex128(dst, src, twiddle, scratch []complex128) bool {
	return mixedRadixForward[complex128](dst, src, twiddle, scratch)
}

func inverseMixedRadixComplex128(dst, src, twiddle, scratch []complex128) bool {
	return mixedRadixInverse[complex128](dst, src, twiddle, scratch)
}

func mixedRadixForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, false)
}

func mixedRadixInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, true)
}

// Recursion hooks for SIMD acceleration.
// By default, these point to the pure Go implementations.
// SIMD-optimized files (like mixedradix_avx2.go) can override these in init().
var (
	recursiveStep64  func(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool)
	recursiveStep128 func(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool)
)

// codeletSchedulable64/128 report whether the installed recursion driver can
// execute a composite radix of the given size directly via a codelet. The
// pure Go driver only knows radices 2/3/4/5, so scheduling any larger radix
// would silently produce garbage (its butterfly switch returns on unknown
// radices). SIMD builds (mixedradix_avx2.go) override these with a predicate
// matching exactly what their recursion hook dispatches.
var (
	codeletSchedulable64  func(int) bool
	codeletSchedulable128 func(int) bool
)

//nolint:gochecknoinits
func init() {
	recursiveStep64 = mixedRadixRecursivePingPongComplex64
	recursiveStep128 = mixedRadixRecursivePingPongComplex128
	codeletSchedulable64 = func(int) bool { return false }
	codeletSchedulable128 = func(int) bool { return false }
}

func mixedRadixTransform[T Complex](dst, src, twiddle, scratch []T, inverse bool) bool {
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

	var (
		radices    [mixedRadixMaxStages]int
		hasCodelet func(int) bool
		zero       T
	)

	// A composite radix may only be scheduled when the recursion driver can
	// dispatch it to a codelet (see codeletSchedulable64/128). Checking the
	// registry alone is not enough: generic (non-SIMD) codelets are registered
	// there too, but the drivers cannot invoke them for sub-transforms.

	switch any(zero).(type) {
	case complex64:
		hasCodelet = codeletSchedulable64
	case complex128:
		hasCodelet = codeletSchedulable128
	default:
		hasCodelet = func(int) bool { return false }
	}

	stageCount := mixedRadixSchedule(n, &radices, hasCodelet)
	if stageCount == 0 {
		return false
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Call through recursion hooks.
	switch any(zero).(type) {
	case complex64:
		recursiveStep64(
			any(work).([]complex64),    //nolint:forcetypeassert
			any(src).([]complex64),     //nolint:forcetypeassert
			any(scratch).([]complex64), //nolint:forcetypeassert
			n, 1, 1, radices[:stageCount],
			any(twiddle).([]complex64), //nolint:forcetypeassert
			inverse,
		)
	case complex128:
		recursiveStep128(
			any(work).([]complex128),    //nolint:forcetypeassert
			any(src).([]complex128),     //nolint:forcetypeassert
			any(scratch).([]complex128), //nolint:forcetypeassert
			n, 1, 1, radices[:stageCount],
			any(twiddle).([]complex128), //nolint:forcetypeassert
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

func mixedRadixSchedule(n int, radices *[mixedRadixMaxStages]int, hasCodelet func(int) bool) int {
	if n < 2 {
		return 0
	}

	count := 0

	// Registry-aware scheduling:
	// If we have a registered codelet for the current size 'n', use it directly!
	// This prevents breaking down large sizes (e.g., 256, 512) into small radices
	// when we have highly optimized AVX2 kernels for them.
	//
	// We verify if *any* codelet exists, not just AVX2, because even a generic
	// codelet for size N might be faster than recursive decomposition.
	//
	// Note: We skip this check for very small sizes (<= 5) as they are handled
	// by the switch statement anyway, and looking them up might be slower.
	if n > 5 && hasCodelet(n) {
		radices[count] = n
		return count + 1
	}

	for n > 1 {
		// Check again at each step: if the remaining size 'n' has a kernel, use it.
		// e.g., 768 = 3 * 256. First loop picks 3. Second loop sees 256.
		// Instead of 256 -> 4*4*4*4, we want 256 directly.
		if n > 5 && hasCodelet(n) {
			radices[count] = n
			count++

			return count
		}

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
			recursiveStep64(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse)
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
			// A radix the driver cannot execute means the scheduler and the
			// recursion hook disagree — a programming error, never a runtime
			// input error. Returning here would leave dst partially written
			// and surface as a wrong answer with a nil error.
			panic("algofft: mixed-radix driver cannot execute radix " + strconv.Itoa(radix) +
				" (scheduler/driver contract violation)")
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
			recursiveStep128(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse)
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
			// A radix the driver cannot execute means the scheduler and the
			// recursion hook disagree — a programming error, never a runtime
			// input error. Returning here would leave dst partially written
			// and surface as a wrong answer with a nil error.
			panic("algofft: mixed-radix driver cannot execute radix " + strconv.Itoa(radix) +
				" (scheduler/driver contract violation)")
		}
	}
}

