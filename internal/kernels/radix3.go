package kernels

import "math"

// Precomputed radix-3 butterfly constants.
// These are used in the radix-3 DFT butterfly operations to avoid
// recomputing them on every call.
//
// For forward transform:
//
//	half = -0.5 + 0i
//	coef = 0 - i*sqrt(3)/2
//
// For inverse transform:
//
//	half = -0.5 + 0i  (same as forward)
//	coef = 0 + i*sqrt(3)/2  (conjugate of forward)
//
//nolint:gochecknoglobals
var (
	radix3Half64     = complex64(-0.5 + 0i)
	radix3CoefFwd64  = complex64(0 - 1i*complex(float32(math.Sqrt(3)/2), 0))
	radix3CoefInv64  = complex64(0 + 1i*complex(float32(math.Sqrt(3)/2), 0))
	radix3Half128    = complex128(-0.5 + 0i)
	radix3CoefFwd128 = complex128(0 - 1i*complex(math.Sqrt(3)/2, 0))
	radix3CoefInv128 = complex128(0 + 1i*complex(math.Sqrt(3)/2, 0))
)

func forwardRadix3Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix3TransformComplex64(dst, src, twiddle, scratch, bitrev, false)
}

func inverseRadix3Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix3TransformComplex64(dst, src, twiddle, scratch, bitrev, true)
}

func forwardRadix3Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix3Forward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix3Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix3Inverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func radix3Forward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix3Transform(dst, src, twiddle, scratch, bitrev, false)
}

func radix3Inverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix3Transform(dst, src, twiddle, scratch, bitrev, true)
}

func radix3Transform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
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

	for size := 3; size <= n; size *= 3 {
		third := size / 3

		step := n / size
		for base := 0; base < n; base += size {
			for j := range third {
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

				var y0, y1, y2 T
				if inverse {
					y0, y1, y2 = butterfly3Inverse(a0, a1, a2)
				} else {
					y0, y1, y2 = butterfly3Forward(a0, a1, a2)
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
		scale := complexFromFloat64[T](1.0/float64(n), 0)
		for i := range dst {
			dst[i] *= scale
		}
	}

	return true
}

// Type-specific butterfly functions to avoid generic overhead

func butterfly3ForwardComplex64(a0, a1, a2 complex64) (complex64, complex64, complex64) {
	t1 := a1 + a2
	t2 := a1 - a2

	y0 := a0 + t1
	base := a0 + radix3Half64*t1
	y1 := base + radix3CoefFwd64*t2
	y2 := base - radix3CoefFwd64*t2

	return y0, y1, y2
}

func butterfly3InverseComplex64(a0, a1, a2 complex64) (complex64, complex64, complex64) {
	t1 := a1 + a2
	t2 := a1 - a2

	y0 := a0 + t1
	base := a0 + radix3Half64*t1
	y1 := base + radix3CoefInv64*t2
	y2 := base - radix3CoefInv64*t2

	return y0, y1, y2
}

func butterfly3ForwardComplex128(a0, a1, a2 complex128) (complex128, complex128, complex128) {
	t1 := a1 + a2
	t2 := a1 - a2

	y0 := a0 + t1
	base := a0 + radix3Half128*t1
	y1 := base + radix3CoefFwd128*t2
	y2 := base - radix3CoefFwd128*t2

	return y0, y1, y2
}

func butterfly3InverseComplex128(a0, a1, a2 complex128) (complex128, complex128, complex128) {
	t1 := a1 + a2
	t2 := a1 - a2

	y0 := a0 + t1
	base := a0 + radix3Half128*t1
	y1 := base + radix3CoefInv128*t2
	y2 := base - radix3CoefInv128*t2

	return y0, y1, y2
}

// Generic wrapper that dispatches to type-specific implementations.
func butterfly3Forward[T Complex](a0, a1, a2 T) (T, T, T) {
	var zero T
	switch any(zero).(type) {
	case complex64:
		y0, y1, y2 := butterfly3ForwardComplex64(
			any(a0).(complex64),
			any(a1).(complex64),
			any(a2).(complex64),
		)

		return any(y0).(T), any(y1).(T), any(y2).(T) //nolint:forcetypeassert
	case complex128:
		y0, y1, y2 := butterfly3ForwardComplex128(
			any(a0).(complex128),
			any(a1).(complex128),
			any(a2).(complex128),
		)

		return any(y0).(T), any(y1).(T), any(y2).(T) //nolint:forcetypeassert
	default:
		panic("unsupported complex type")
	}
}

func butterfly3Inverse[T Complex](a0, a1, a2 T) (T, T, T) {
	var zero T
	switch any(zero).(type) {
	case complex64:
		y0, y1, y2 := butterfly3InverseComplex64(
			any(a0).(complex64),
			any(a1).(complex64),
			any(a2).(complex64),
		)

		return any(y0).(T), any(y1).(T), any(y2).(T) //nolint:forcetypeassert
	case complex128:
		y0, y1, y2 := butterfly3InverseComplex128(
			any(a0).(complex128),
			any(a1).(complex128),
			any(a2).(complex128),
		)

		return any(y0).(T), any(y1).(T), any(y2).(T) //nolint:forcetypeassert
	default:
		panic("unsupported complex type")
	}
}

// Public exports for internal/fft - generic wrappers.
func Butterfly3Forward[T Complex](a0, a1, a2 T) (T, T, T) {
	return butterfly3Forward(a0, a1, a2)
}

func Butterfly3Inverse[T Complex](a0, a1, a2 T) (T, T, T) {
	return butterfly3Inverse(a0, a1, a2)
}

// Public exports for internal/fft - type-specific functions for direct calls.
func Butterfly3ForwardComplex64(a0, a1, a2 complex64) (complex64, complex64, complex64) {
	return butterfly3ForwardComplex64(a0, a1, a2)
}

func Butterfly3InverseComplex64(a0, a1, a2 complex64) (complex64, complex64, complex64) {
	return butterfly3InverseComplex64(a0, a1, a2)
}

func Butterfly3ForwardComplex128(a0, a1, a2 complex128) (complex128, complex128, complex128) {
	return butterfly3ForwardComplex128(a0, a1, a2)
}

func Butterfly3InverseComplex128(a0, a1, a2 complex128) (complex128, complex128, complex128) {
	return butterfly3InverseComplex128(a0, a1, a2)
}

func reverseBase3(x, digits int) int {
	result := 0
	for range digits {
		result = result*3 + (x % 3)
		x /= 3
	}

	return result
}

func logBase3(n int) int {
	result := 0

	for n > 1 {
		n /= 3
		result++
	}

	return result
}
