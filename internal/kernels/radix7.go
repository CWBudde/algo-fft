package kernels

import "math"

// radix7Cos64/128 and radix7Sin64/128 hold the three conjugate-pair
// coefficients ck = cos(2*pi*k/7), sk = sin(2*pi*k/7), k = 1..3, used by the
// pair-form butterfly below. See the derivation in
// internal/asm/amd64/avx2_f32_mixedradix_stage7.s ("RADIX-7 BUTTERFLY"),
// which this Go implementation mirrors term for term.
//
//nolint:gochecknoglobals
var (
	radix7Cos64  [3]float32
	radix7Sin64  [3]float32
	radix7Cos128 [3]float64
	radix7Sin128 [3]float64
)

//nolint:gochecknoinits
func init() {
	for k := 1; k <= 3; k++ {
		angle := 2 * math.Pi * float64(k) / 7
		c := math.Cos(angle)
		s := math.Sin(angle)
		radix7Cos128[k-1] = c
		radix7Sin128[k-1] = s
		radix7Cos64[k-1] = float32(c)
		radix7Sin64[k-1] = float32(s)
	}
}

// Butterfly7ForwardComplex64 applies the forward radix-7 DFT butterfly to a
// in place. Inputs must already carry their stage twiddle factors.
func Butterfly7ForwardComplex64(a *[7]complex64) {
	y0, m1, q1, m2, q2, m3, q3 := butterfly7CoreComplex64(a)

	a[0] = y0
	a[1] = m1 + q1
	a[6] = m1 - q1
	a[2] = m2 + q2
	a[5] = m2 - q2
	a[3] = m3 + q3
	a[4] = m3 - q3
}

// Butterfly7InverseComplex64 applies the inverse (conjugate) radix-7 DFT
// butterfly to a in place. No 1/7 scaling is applied.
func Butterfly7InverseComplex64(a *[7]complex64) {
	// The inverse butterfly replaces every -i with +i, i.e. q_inv = -q_fwd,
	// so it is the forward core with the +q/-q outputs swapped.
	y0, m1, q1, m2, q2, m3, q3 := butterfly7CoreComplex64(a)

	a[0] = y0
	a[1] = m1 - q1
	a[6] = m1 + q1
	a[2] = m2 - q2
	a[5] = m2 + q2
	a[3] = m3 - q3
	a[4] = m3 + q3
}

// butterfly7CoreComplex64 computes the direction-independent half of the
// radix-7 conjugate-pair butterfly: y0 and the three (m, q) pairs, with q
// formed via the forward -i factor.
//
// Conjugate-pair form: with wk = W^k and w(7-k) = conj(wk), the direct 7x7
// matrix product (42 complex multiplies) collapses to 6 real-by-complex
// multiplies (12 real multiplies) plus adds. c1..c3, s1..s3 are real
// scalars, so c*t is computed component-wise rather than through a
// complex-by-complex multiply. See avx2_f32_mixedradix_stage7.s.
func butterfly7CoreComplex64(a *[7]complex64) (y0, m1, q1, m2, q2, m3, q3 complex64) {
	c1, c2, c3 := radix7Cos64[0], radix7Cos64[1], radix7Cos64[2]
	s1, s2, s3 := radix7Sin64[0], radix7Sin64[1], radix7Sin64[2]

	t1 := a[1] + a[6]
	u1 := a[1] - a[6]
	t2 := a[2] + a[5]
	u2 := a[2] - a[5]
	t3 := a[3] + a[4]
	u3 := a[3] - a[4]

	y0 = a[0] + t1 + t2 + t3

	t1r, t1i := real(t1), imag(t1)
	t2r, t2i := real(t2), imag(t2)
	t3r, t3i := real(t3), imag(t3)
	u1r, u1i := real(u1), imag(u1)
	u2r, u2i := real(u2), imag(u2)
	u3r, u3i := real(u3), imag(u3)

	m1 = a[0] + complex(c1*t1r+c2*t2r+c3*t3r, c1*t1i+c2*t2i+c3*t3i)
	m2 = a[0] + complex(c2*t1r+c3*t2r+c1*t3r, c2*t1i+c3*t2i+c1*t3i)
	m3 = a[0] + complex(c3*t1r+c1*t2r+c2*t3r, c3*t1i+c1*t2i+c2*t3i)

	// q = -i * (sum): -i*(x+iy) = y - i*x.
	sum1r := s1*u1r + s2*u2r + s3*u3r
	sum1i := s1*u1i + s2*u2i + s3*u3i
	q1 = complex(sum1i, -sum1r)

	sum2r := s2*u1r - s3*u2r - s1*u3r
	sum2i := s2*u1i - s3*u2i - s1*u3i
	q2 = complex(sum2i, -sum2r)

	sum3r := s3*u1r - s1*u2r + s2*u3r
	sum3i := s3*u1i - s1*u2i + s2*u3i
	q3 = complex(sum3i, -sum3r)

	return y0, m1, q1, m2, q2, m3, q3
}
