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

// Recursion hooks for SIMD acceleration.
// By default, these point to the pure Go implementations.
// SIMD-optimized files (like mixedradix_avx2.go) can override these in init().
var (
	recursiveStep64  func(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool)
	recursiveStep128 func(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool)
)

//nolint:gochecknoinits
func init() {
	recursiveStep64 = mixedRadixRecursivePingPongComplex64
	recursiveStep128 = mixedRadixRecursivePingPongComplex128
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

	var (
		radices    [mixedRadixMaxStages]int
		hasCodelet func(int) bool
		zero       T
	)

	// Determine which registry to check based on type T

	switch any(zero).(type) {
	case complex64:
		hasCodelet = kernels.Registry64.Has
	case complex128:
		hasCodelet = kernels.Registry128.Has
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

// mixedRadixPermutation computes the data reordering index for the iterative
// mixed-radix FFT decomposition.
//
// This function maps input index x to its position in the decomposed data layout.
// The decomposition pattern matches the recursive algorithm's implicit reordering:
// for each stage, data is grouped by extracting digits in the mixed-radix
// representation and using them as offsets with decreasing strides.
//
// Example: n=6, radices=[3,2]
//
//	x=1 → digits (d0=1, d1=0) → result = 1*2 + 0*1 = 2
//	x=3 → digits (d0=0, d1=1) → result = 0*2 + 1*1 = 1
//	Permutation: [0,2,4,1,3,5]
func mixedRadixPermutation(x, n int, radices []int, count int) int {
	// Extract digits in forward order: d[i] = (x / prod(radices[0..i-1])) % radices[i]
	var digits [mixedRadixMaxStages]int

	temp := x
	for i := range count {
		digits[i] = temp % radices[i]
		temp /= radices[i]
	}

	// Calculate result: sum of d[i] * span[i]
	// where span[i] = n / (radices[0] * radices[1] * ... * radices[i])
	result := 0

	product := 1
	for i := range count {
		product *= radices[i]
		span := n / product
		result += digits[i] * span
	}

	return result
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

// ==============================================================================
// WORK IN PROGRESS: Iterative Mixed-Radix Implementation
// ==============================================================================
//
// The functions below implement an iterative alternative to the recursive
// mixed-radix FFT. They are currently DISABLED due to correctness issues.
//
// Status: The permutation logic and stage processing are implemented but tests
// fail, indicating the data flow pattern doesn't match the recursive version.
//
// Challenges identified:
// 1. The recursive decomposition has a complex depth-first data access pattern
// 2. Simple pre-permutation + sequential stage processing doesn't capture this
// 3. Buffer management and stage ordering need more investigation
//
// Future work: Consider alternative iterative algorithms (four-step, Bluestein)
// or deeper analysis of the recursive data flow to find correct mapping.
// ==============================================================================

// mixedRadixIterativeComplex64 implements mixed-radix FFT using an iterative
// approach instead of recursion, eliminating function call overhead and reducing
// stack usage from O(log n) to O(1).
//
// This implementation achieves the same result as the recursive version by:
//  1. Pre-permuting the input data using mixed-radix decomposition pattern
//  2. Processing butterfly stages sequentially with explicit loops
//  3. Using ping-pong buffering to eliminate intermediate copies
//
// Parameters match the recursive version for drop-in replacement.
//
// NOTE: Currently disabled - see WIP comment above.
func mixedRadixIterativeComplex64(dst, src, scratch []complex64, n int, radices []int, twiddle []complex64, inverse bool) {
	stageCount := len(radices)

	// Stage metadata structure
	type stage struct {
		radix int
		span  int
		step  int
	}

	// Precompute stage schedule
	var stages [mixedRadixMaxStages]stage

	stride := 1
	step := n

	for i := range stageCount {
		radix := radices[i]
		stages[i].radix = radix
		stages[i].span = n / (stride * radix)
		step /= radix
		stages[i].step = step
		stride *= radix
	}

	// Pre-permute input data using mixed-radix decomposition pattern
	for i := range n {
		j := mixedRadixPermutation(i, n, radices, stageCount)
		dst[j] = src[i]
	}

	// Set up ping-pong buffers
	inputBuf := dst
	outputBuf := scratch

	// Iterative butterfly stages
	// Process stages in REVERSE order to match recursive decomposition
	// (recursive version applies butterflies from innermost to outermost radix)
	for stageIdx := stageCount - 1; stageIdx >= 0; stageIdx-- {
		s := stages[stageIdx]
		radix := s.radix
		span := s.span
		step := s.step
		groupCount := n / (span * radix)

		// Radix-specific dispatch with separate loops per radix for performance
		switch radix {
		case 2:
			// Radix-2 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 2
				for k := range span {
					idx := baseIdx + k

					w1 := twiddle[k*step]
					if inverse {
						w1 = conj(w1)
					}

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]

					outputBuf[idx] = a0 + a1
					outputBuf[idx+span] = a0 - a1
				}
			}

		case 3:
			// Radix-3 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 3
				for k := range span {
					idx := baseIdx + k
					w1 := twiddle[k*step]
					w2 := twiddle[2*k*step]

					if inverse {
						w1 = conj(w1)
						w2 = conj(w2)
					}

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]
					a2 := w2 * inputBuf[idx+2*span]

					var y0, y1, y2 complex64
					if inverse {
						y0, y1, y2 = kernels.Butterfly3InverseComplex64(a0, a1, a2)
					} else {
						y0, y1, y2 = kernels.Butterfly3ForwardComplex64(a0, a1, a2)
					}

					outputBuf[idx] = y0
					outputBuf[idx+span] = y1
					outputBuf[idx+2*span] = y2
				}
			}

		case 4:
			// Radix-4 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 4
				for k := range span {
					idx := baseIdx + k
					w1 := twiddle[k*step]
					w2 := twiddle[2*k*step]
					w3 := twiddle[3*k*step]

					if inverse {
						w1 = conj(w1)
						w2 = conj(w2)
						w3 = conj(w3)
					}

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]
					a2 := w2 * inputBuf[idx+2*span]
					a3 := w3 * inputBuf[idx+3*span]

					var y0, y1, y2, y3 complex64
					if inverse {
						y0, y1, y2, y3 = kernels.Butterfly4InverseComplex64(a0, a1, a2, a3)
					} else {
						y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex64(a0, a1, a2, a3)
					}

					outputBuf[idx] = y0
					outputBuf[idx+span] = y1
					outputBuf[idx+2*span] = y2
					outputBuf[idx+3*span] = y3
				}
			}

		case 5:
			// Radix-5 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 5
				for k := range span {
					idx := baseIdx + k
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

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]
					a2 := w2 * inputBuf[idx+2*span]
					a3 := w3 * inputBuf[idx+3*span]
					a4 := w4 * inputBuf[idx+4*span]

					var y0, y1, y2, y3, y4 complex64
					if inverse {
						y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex64(a0, a1, a2, a3, a4)
					} else {
						y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
					}

					outputBuf[idx] = y0
					outputBuf[idx+span] = y1
					outputBuf[idx+2*span] = y2
					outputBuf[idx+3*span] = y3
					outputBuf[idx+4*span] = y4
				}
			}
		}

		// Swap buffers for next stage
		inputBuf, outputBuf = outputBuf, inputBuf
	}

	// After stageCount swaps:
	//   - even stageCount: result in dst
	//   - odd stageCount: result in scratch
	if stageCount%2 == 1 {
		copy(dst, scratch[:n])
	}
}

// mixedRadixIterativeComplex128 implements mixed-radix FFT using an iterative
// approach instead of recursion, eliminating function call overhead and reducing
// stack usage from O(log n) to O(1).
//
// This is the complex128 version of mixedRadixIterativeComplex64.
// See mixedRadixIterativeComplex64 for implementation details.
func mixedRadixIterativeComplex128(dst, src, scratch []complex128, n int, radices []int, twiddle []complex128, inverse bool) {
	stageCount := len(radices)

	// Stage metadata structure
	type stage struct {
		radix int
		span  int
		step  int
	}

	// Precompute stage schedule
	var stages [mixedRadixMaxStages]stage

	stride := 1
	step := n

	for i := range stageCount {
		radix := radices[i]
		stages[i].radix = radix
		stages[i].span = n / (stride * radix)
		step /= radix
		stages[i].step = step
		stride *= radix
	}

	// Pre-permute input data using mixed-radix decomposition pattern
	for i := range n {
		j := mixedRadixPermutation(i, n, radices, stageCount)
		dst[j] = src[i]
	}

	// Set up ping-pong buffers
	inputBuf := dst
	outputBuf := scratch

	// Iterative butterfly stages
	// Process stages in REVERSE order to match recursive decomposition
	// (recursive version applies butterflies from innermost to outermost radix)
	for stageIdx := stageCount - 1; stageIdx >= 0; stageIdx-- {
		s := stages[stageIdx]
		radix := s.radix
		span := s.span
		step := s.step
		groupCount := n / (span * radix)

		// Radix-specific dispatch with separate loops per radix for performance
		switch radix {
		case 2:
			// Radix-2 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 2
				for k := range span {
					idx := baseIdx + k

					w1 := twiddle[k*step]
					if inverse {
						w1 = conj(w1)
					}

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]

					outputBuf[idx] = a0 + a1
					outputBuf[idx+span] = a0 - a1
				}
			}

		case 3:
			// Radix-3 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 3
				for k := range span {
					idx := baseIdx + k
					w1 := twiddle[k*step]
					w2 := twiddle[2*k*step]

					if inverse {
						w1 = conj(w1)
						w2 = conj(w2)
					}

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]
					a2 := w2 * inputBuf[idx+2*span]

					var y0, y1, y2 complex128
					if inverse {
						y0, y1, y2 = kernels.Butterfly3InverseComplex128(a0, a1, a2)
					} else {
						y0, y1, y2 = kernels.Butterfly3ForwardComplex128(a0, a1, a2)
					}

					outputBuf[idx] = y0
					outputBuf[idx+span] = y1
					outputBuf[idx+2*span] = y2
				}
			}

		case 4:
			// Radix-4 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 4
				for k := range span {
					idx := baseIdx + k
					w1 := twiddle[k*step]
					w2 := twiddle[2*k*step]
					w3 := twiddle[3*k*step]

					if inverse {
						w1 = conj(w1)
						w2 = conj(w2)
						w3 = conj(w3)
					}

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]
					a2 := w2 * inputBuf[idx+2*span]
					a3 := w3 * inputBuf[idx+3*span]

					var y0, y1, y2, y3 complex128
					if inverse {
						y0, y1, y2, y3 = kernels.Butterfly4InverseComplex128(a0, a1, a2, a3)
					} else {
						y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex128(a0, a1, a2, a3)
					}

					outputBuf[idx] = y0
					outputBuf[idx+span] = y1
					outputBuf[idx+2*span] = y2
					outputBuf[idx+3*span] = y3
				}
			}

		case 5:
			// Radix-5 butterfly loop
			for group := range groupCount {
				baseIdx := group * span * 5
				for k := range span {
					idx := baseIdx + k
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

					a0 := inputBuf[idx]
					a1 := w1 * inputBuf[idx+span]
					a2 := w2 * inputBuf[idx+2*span]
					a3 := w3 * inputBuf[idx+3*span]
					a4 := w4 * inputBuf[idx+4*span]

					var y0, y1, y2, y3, y4 complex128
					if inverse {
						y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex128(a0, a1, a2, a3, a4)
					} else {
						y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex128(a0, a1, a2, a3, a4)
					}

					outputBuf[idx] = y0
					outputBuf[idx+span] = y1
					outputBuf[idx+2*span] = y2
					outputBuf[idx+3*span] = y3
					outputBuf[idx+4*span] = y4
				}
			}
		}

		// Swap buffers for next stage
		inputBuf, outputBuf = outputBuf, inputBuf
	}

	// After stageCount swaps:
	//   - even stageCount: result in dst
	//   - odd stageCount: result in scratch
	if stageCount%2 == 1 {
		copy(dst, scratch[:n])
	}
}
