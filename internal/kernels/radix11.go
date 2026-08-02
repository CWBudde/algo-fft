package kernels

import "math"

// radix11Cos64/128 and radix11Sin64/128 hold the five conjugate-pair
// coefficients ck = cos(2*pi*k/11), sk = sin(2*pi*k/11), k = 1..5, used by
// the pair-form butterfly below. See the derivation in
// internal/asm/amd64/avx2_f32_mixedradix_stage11.s ("RADIX-11 BUTTERFLY"),
// which this Go implementation mirrors term for term (that header notes it
// was verified against the previous 11x11-matrix form of this butterfly).
//
//nolint:gochecknoglobals
var (
	radix11Cos64  [5]float32
	radix11Sin64  [5]float32
	radix11Cos128 [5]float64
	radix11Sin128 [5]float64
)

//nolint:gochecknoinits
func init() {
	for k := 1; k <= 5; k++ {
		angle := 2 * math.Pi * float64(k) / 11
		c := math.Cos(angle)
		s := math.Sin(angle)
		radix11Cos128[k-1] = c
		radix11Sin128[k-1] = s
		radix11Cos64[k-1] = float32(c)
		radix11Sin64[k-1] = float32(s)
	}
}

// Butterfly11ForwardComplex64 applies the forward radix-11 DFT butterfly to a
// in place. Inputs must already carry their stage twiddle factors.
func Butterfly11ForwardComplex64(a *[11]complex64) {
	y0, m1, q1, m2, q2, m3, q3, m4, q4, m5, q5 := butterfly11CoreComplex64(a)

	a[0] = y0
	a[1] = m1 + q1
	a[10] = m1 - q1
	a[2] = m2 + q2
	a[9] = m2 - q2
	a[3] = m3 + q3
	a[8] = m3 - q3
	a[4] = m4 + q4
	a[7] = m4 - q4
	a[5] = m5 + q5
	a[6] = m5 - q5
}

// Butterfly11InverseComplex64 applies the inverse (conjugate) radix-11 DFT
// butterfly to a in place. No 1/11 scaling is applied.
func Butterfly11InverseComplex64(a *[11]complex64) {
	// The inverse butterfly replaces every -i with +i, i.e. q_inv = -q_fwd,
	// so it is the forward core with the +q/-q outputs swapped.
	y0, m1, q1, m2, q2, m3, q3, m4, q4, m5, q5 := butterfly11CoreComplex64(a)

	a[0] = y0
	a[1] = m1 - q1
	a[10] = m1 + q1
	a[2] = m2 - q2
	a[9] = m2 + q2
	a[3] = m3 - q3
	a[8] = m3 + q3
	a[4] = m4 - q4
	a[7] = m4 + q4
	a[5] = m5 - q5
	a[6] = m5 + q5
}

// butterfly11CoreComplex64 computes the direction-independent half of the
// radix-11 conjugate-pair butterfly: y0 and the five (m, q) pairs, with q
// formed via the forward -i factor.
//
// Conjugate-pair form: with wk = W^k and w(11-k) = conj(wk), the direct
// 11x11 matrix product (100 complex multiplies) collapses to 10 real-by-
// complex multiplies (20 real multiplies) plus adds. c1..c5, s1..s5 are real
// scalars, so c*t is computed component-wise rather than through a
// complex-by-complex multiply. See avx2_f32_mixedradix_stage11.s.
func butterfly11CoreComplex64(a *[11]complex64) (y0, m1, q1, m2, q2, m3, q3, m4, q4, m5, q5 complex64) {
	c1, c2, c3, c4, c5 := radix11Cos64[0], radix11Cos64[1], radix11Cos64[2], radix11Cos64[3], radix11Cos64[4]
	s1, s2, s3, s4, s5 := radix11Sin64[0], radix11Sin64[1], radix11Sin64[2], radix11Sin64[3], radix11Sin64[4]

	t1 := a[1] + a[10]
	u1 := a[1] - a[10]
	t2 := a[2] + a[9]
	u2 := a[2] - a[9]
	t3 := a[3] + a[8]
	u3 := a[3] - a[8]
	t4 := a[4] + a[7]
	u4 := a[4] - a[7]
	t5 := a[5] + a[6]
	u5 := a[5] - a[6]

	y0 = a[0] + t1 + t2 + t3 + t4 + t5

	t1r, t1i := real(t1), imag(t1)
	t2r, t2i := real(t2), imag(t2)
	t3r, t3i := real(t3), imag(t3)
	t4r, t4i := real(t4), imag(t4)
	t5r, t5i := real(t5), imag(t5)
	u1r, u1i := real(u1), imag(u1)
	u2r, u2i := real(u2), imag(u2)
	u3r, u3i := real(u3), imag(u3)
	u4r, u4i := real(u4), imag(u4)
	u5r, u5i := real(u5), imag(u5)

	m1 = a[0] + complex(
		c1*t1r+c2*t2r+c3*t3r+c4*t4r+c5*t5r,
		c1*t1i+c2*t2i+c3*t3i+c4*t4i+c5*t5i,
	)
	m2 = a[0] + complex(
		c2*t1r+c4*t2r+c5*t3r+c3*t4r+c1*t5r,
		c2*t1i+c4*t2i+c5*t3i+c3*t4i+c1*t5i,
	)
	m3 = a[0] + complex(
		c3*t1r+c5*t2r+c2*t3r+c1*t4r+c4*t5r,
		c3*t1i+c5*t2i+c2*t3i+c1*t4i+c4*t5i,
	)
	m4 = a[0] + complex(
		c4*t1r+c3*t2r+c1*t3r+c5*t4r+c2*t5r,
		c4*t1i+c3*t2i+c1*t3i+c5*t4i+c2*t5i,
	)
	m5 = a[0] + complex(
		c5*t1r+c1*t2r+c4*t3r+c2*t4r+c3*t5r,
		c5*t1i+c1*t2i+c4*t3i+c2*t4i+c3*t5i,
	)

	// q = -i * (sum): -i*(x+iy) = y - i*x.
	sum1r := s1*u1r + s2*u2r + s3*u3r + s4*u4r + s5*u5r
	sum1i := s1*u1i + s2*u2i + s3*u3i + s4*u4i + s5*u5i
	q1 = complex(sum1i, -sum1r)

	sum2r := s2*u1r + s4*u2r - s5*u3r - s3*u4r - s1*u5r
	sum2i := s2*u1i + s4*u2i - s5*u3i - s3*u4i - s1*u5i
	q2 = complex(sum2i, -sum2r)

	sum3r := s3*u1r - s5*u2r - s2*u3r + s1*u4r + s4*u5r
	sum3i := s3*u1i - s5*u2i - s2*u3i + s1*u4i + s4*u5i
	q3 = complex(sum3i, -sum3r)

	sum4r := s4*u1r - s3*u2r + s1*u3r + s5*u4r - s2*u5r
	sum4i := s4*u1i - s3*u2i + s1*u3i + s5*u4i - s2*u5i
	q4 = complex(sum4i, -sum4r)

	sum5r := s5*u1r - s1*u2r + s4*u3r - s2*u4r + s3*u5r
	sum5i := s5*u1i - s1*u2i + s4*u3i - s2*u4i + s3*u5i
	q5 = complex(sum5i, -sum5r)

	return y0, m1, q1, m2, q2, m3, q3, m4, q4, m5, q5
}
