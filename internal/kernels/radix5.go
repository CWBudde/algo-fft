package kernels

import "math"

// radix5Cos64/128 and radix5Sin64/128 hold the two conjugate-pair
// coefficients c1 = cos(2*pi/5), c2 = cos(4*pi/5), s1 = sin(2*pi/5),
// s2 = sin(4*pi/5) used by the pair-form butterfly below. See the derivation
// in internal/asm/amd64/avx2_f32_mixedradix_stage5.s ("RADIX-5 BUTTERFLY"),
// which this Go implementation mirrors term for term.
//
//nolint:gochecknoglobals
var (
	radix5Cos64  [2]float32
	radix5Sin64  [2]float32
	radix5Cos128 [2]float64
	radix5Sin128 [2]float64
)

//nolint:gochecknoinits
func init() {
	for k := 1; k <= 2; k++ {
		angle := 2 * math.Pi * float64(k) / 5
		c := math.Cos(angle)
		s := math.Sin(angle)
		radix5Cos128[k-1] = c
		radix5Sin128[k-1] = s
		radix5Cos64[k-1] = float32(c)
		radix5Sin64[k-1] = float32(s)
	}
}

func forwardRadix5Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix5TransformComplex64(dst, src, twiddle, scratch, bitrev, false)
}

func inverseRadix5Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return radix5TransformComplex64(dst, src, twiddle, scratch, bitrev, true)
}

func forwardRadix5Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix5Forward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseRadix5Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	return radix5Inverse[complex128](dst, src, twiddle, scratch, bitrev)
}

func radix5Forward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix5Transform(dst, src, twiddle, scratch, bitrev, false)
}

func radix5Inverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return radix5Transform(dst, src, twiddle, scratch, bitrev, true)
}

func radix5Transform[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
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

	for size := 5; size <= n; size *= 5 {
		span := size / 5

		step := n / size
		for base := 0; base < n; base += size {
			for j := range span {
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

				var y0, y1, y2, y3, y4 T
				if inverse {
					y0, y1, y2, y3, y4 = butterfly5Inverse(a0, a1, a2, a3, a4)
				} else {
					y0, y1, y2, y3, y4 = butterfly5Forward(a0, a1, a2, a3, a4)
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
		scale := complexFromFloat64[T](1.0/float64(n), 0)
		for i := range dst {
			dst[i] *= scale
		}
	}

	return true
}

// Type-specific butterfly functions to avoid generic overhead.
//
// Conjugate-pair form: with w1 = W^1, w4 = conj(w1), w2 = W^2, w3 = conj(w2),
// the direct 5x5 matrix product (20 complex multiplies) collapses to 4
// real-by-complex multiplies (8 real multiplies) plus adds. c1, c2, s1, s2
// are real scalars, so c*t is computed component-wise rather than through a
// complex-by-complex multiply. See avx2_f32_mixedradix_stage5.s.
//
// butterfly5CoreComplex64 computes the direction-independent half: y0 and
// the two (m, q) pairs, with q formed via the forward -i factor. The
// direction only changes which of m+q / m-q lands on which output index (see
// butterfly5ForwardComplex64 / butterfly5InverseComplex64), so both
// directions share this one arithmetic body.
func butterfly5CoreComplex64(a0, a1, a2, a3, a4 complex64) (y0, m1, q1, m2, q2 complex64) {
	c1, c2 := radix5Cos64[0], radix5Cos64[1]
	s1, s2 := radix5Sin64[0], radix5Sin64[1]

	t1 := a1 + a4
	t2 := a2 + a3
	t3 := a1 - a4
	t4 := a2 - a3

	y0 = a0 + t1 + t2

	t1r, t1i := real(t1), imag(t1)
	t2r, t2i := real(t2), imag(t2)
	t3r, t3i := real(t3), imag(t3)
	t4r, t4i := real(t4), imag(t4)

	m1 = a0 + complex(c1*t1r+c2*t2r, c1*t1i+c2*t2i)
	m2 = a0 + complex(c2*t1r+c1*t2r, c2*t1i+c1*t2i)

	// q = -i * (s1*t3 + s2*t4): -i*(x+iy) = y - i*x.
	sum1r := s1*t3r + s2*t4r
	sum1i := s1*t3i + s2*t4i
	q1 = complex(sum1i, -sum1r)

	sum2r := s2*t3r - s1*t4r
	sum2i := s2*t3i - s1*t4i
	q2 = complex(sum2i, -sum2r)

	return y0, m1, q1, m2, q2
}

func butterfly5ForwardComplex64(a0, a1, a2, a3, a4 complex64) (complex64, complex64, complex64, complex64, complex64) {
	y0, m1, q1, m2, q2 := butterfly5CoreComplex64(a0, a1, a2, a3, a4)

	return y0, m1 + q1, m2 + q2, m2 - q2, m1 - q1
}

func butterfly5InverseComplex64(a0, a1, a2, a3, a4 complex64) (complex64, complex64, complex64, complex64, complex64) {
	// The inverse butterfly replaces every -i with +i, i.e. q_inv = -q_fwd,
	// so it is the forward core with the +q/-q outputs swapped.
	y0, m1, q1, m2, q2 := butterfly5CoreComplex64(a0, a1, a2, a3, a4)

	return y0, m1 - q1, m2 - q2, m2 + q2, m1 + q1
}

// Generic wrapper that dispatches to type-specific implementations.
func butterfly5Forward[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	switch a0v := any(a0).(type) {
	case complex64:
		a1v, _ := any(a1).(complex64)
		a2v, _ := any(a2).(complex64)
		a3v, _ := any(a3).(complex64)
		a4v, _ := any(a4).(complex64)
		y0, y1, y2, y3, y4 := butterfly5ForwardComplex64(a0v, a1v, a2v, a3v, a4v)
		r0, _ := any(y0).(T)
		r1, _ := any(y1).(T)
		r2, _ := any(y2).(T)
		r3, _ := any(y3).(T)
		r4, _ := any(y4).(T)

		return r0, r1, r2, r3, r4
	case complex128:
		a1v, _ := any(a1).(complex128)
		a2v, _ := any(a2).(complex128)
		a3v, _ := any(a3).(complex128)
		a4v, _ := any(a4).(complex128)
		y0, y1, y2, y3, y4 := butterfly5ForwardComplex128(a0v, a1v, a2v, a3v, a4v)
		r0, _ := any(y0).(T)
		r1, _ := any(y1).(T)
		r2, _ := any(y2).(T)
		r3, _ := any(y3).(T)
		r4, _ := any(y4).(T)

		return r0, r1, r2, r3, r4
	default:
		panic("unsupported complex type")
	}
}

func butterfly5Inverse[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	switch a0v := any(a0).(type) {
	case complex64:
		a1v, _ := any(a1).(complex64)
		a2v, _ := any(a2).(complex64)
		a3v, _ := any(a3).(complex64)
		a4v, _ := any(a4).(complex64)
		y0, y1, y2, y3, y4 := butterfly5InverseComplex64(a0v, a1v, a2v, a3v, a4v)
		r0, _ := any(y0).(T)
		r1, _ := any(y1).(T)
		r2, _ := any(y2).(T)
		r3, _ := any(y3).(T)
		r4, _ := any(y4).(T)

		return r0, r1, r2, r3, r4
	case complex128:
		a1v, _ := any(a1).(complex128)
		a2v, _ := any(a2).(complex128)
		a3v, _ := any(a3).(complex128)
		a4v, _ := any(a4).(complex128)
		y0, y1, y2, y3, y4 := butterfly5InverseComplex128(a0v, a1v, a2v, a3v, a4v)
		r0, _ := any(y0).(T)
		r1, _ := any(y1).(T)
		r2, _ := any(y2).(T)
		r3, _ := any(y3).(T)
		r4, _ := any(y4).(T)

		return r0, r1, r2, r3, r4
	default:
		panic("unsupported complex type")
	}
}

// Public exports for internal/fft - generic wrappers.
func Butterfly5Forward[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	return butterfly5Forward(a0, a1, a2, a3, a4)
}

func Butterfly5Inverse[T Complex](a0, a1, a2, a3, a4 T) (T, T, T, T, T) {
	return butterfly5Inverse(a0, a1, a2, a3, a4)
}

// Public exports for internal/fft - type-specific functions for direct calls.
func Butterfly5ForwardComplex64(a0, a1, a2, a3, a4 complex64) (complex64, complex64, complex64, complex64, complex64) {
	return butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
}

func Butterfly5InverseComplex64(a0, a1, a2, a3, a4 complex64) (complex64, complex64, complex64, complex64, complex64) {
	return butterfly5InverseComplex64(a0, a1, a2, a3, a4)
}

func reverseBase5(x, digits int) int {
	result := 0
	for range digits {
		result = result*5 + (x % 5)
		x /= 5
	}

	return result
}

func logBase5(n int) int {
	result := 0

	for n > 1 {
		n /= 5
		result++
	}

	return result
}
