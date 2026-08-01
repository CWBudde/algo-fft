package kernels

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// naiveDFT16 is the O(n^2) definition, evaluated in float64 so it can serve as
// ground truth for a float32 kernel. sign is -1 for the forward transform and
// +1 for the unnormalised inverse.
func naiveDFT16(x *[16]complex64, sign float64) [16]complex128 {
	var out [16]complex128

	for k := range 16 {
		var acc complex128

		for n := range 16 {
			theta := sign * 2 * math.Pi * float64(k*n) / 16
			acc += complex(float64(real(x[n])), float64(imag(x[n]))) * cmplx.Exp(complex(0, theta))
		}

		out[k] = acc
	}

	return out
}

func randomBlock16(seed uint64) [16]complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))

	var x [16]complex64
	for i := range x {
		x[i] = complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
	}

	return x
}

func maxAbsDiff16(got *[16]complex64, want [16]complex128) float64 {
	worst := 0.0

	for i := range got {
		d := cmplx.Abs(complex(float64(real(got[i])), float64(imag(got[i]))) - want[i])
		if d > worst {
			worst = d
		}
	}

	return worst
}

// The factored butterfly's twiddle layer is the only arithmetic not already
// covered by the radix-4 tests, and a wrong exponent there produces a spectrum
// that still satisfies Parseval and linearity. Comparing every bin against the
// definition, over random input, is what actually catches it.
func TestButterfly16ForwardMatchesReference(t *testing.T) {
	t.Parallel()

	for seed := range uint64(32) {
		x := randomBlock16(seed)
		want := naiveDFT16(&x, -1)

		got := x
		butterfly16ForwardComplex64(&got)

		if d := maxAbsDiff16(&got, want); d > 1e-4 {
			t.Fatalf("seed %d: forward max |diff| = %g", seed, d)
		}
	}
}

func TestButterfly16InverseMatchesReference(t *testing.T) {
	t.Parallel()

	for seed := range uint64(32) {
		x := randomBlock16(seed + 1000)
		want := naiveDFT16(&x, +1)

		got := x
		butterfly16InverseComplex64(&got)

		if d := maxAbsDiff16(&got, want); d > 1e-4 {
			t.Fatalf("seed %d: inverse max |diff| = %g", seed, d)
		}
	}
}

func TestButterfly16RoundTrip(t *testing.T) {
	t.Parallel()

	for seed := range uint64(32) {
		x := randomBlock16(seed + 2000)

		got := x
		butterfly16ForwardComplex64(&got)
		butterfly16InverseComplex64(&got)

		for i := range got {
			scaled := complex(real(got[i])/16, imag(got[i])/16)
			if d := cmplx.Abs(complex128(scaled) - complex128(x[i])); d > 1e-4 {
				t.Fatalf("seed %d, bin %d: round-trip |diff| = %g", seed, i, d)
			}
		}
	}
}

// The specialised rotations are the whole reason the factored form is cheap, so
// each one is pinned against a general complex multiply by the same constant.
// A sign slip in mulW16Pow6Complex64 would otherwise only show up as a small
// error in eight of sixteen bins.
func TestW16RotationsMatchGeneralMultiply(t *testing.T) {
	t.Parallel()

	w := func(e int) complex128 {
		return cmplx.Exp(complex(0, -2*math.Pi*float64(e)/16))
	}

	cases := []struct {
		name string
		exp  int
		fn   func(complex64) complex64
	}{
		{"W^1", 1, mulW16Pow1Complex64},
		{"W^2", 2, mulW16Pow2Complex64},
		{"W^3", 3, mulW16Pow3Complex64},
		{"W^4", 4, mulNegIComplex64},
		{"W^6", 6, mulW16Pow6Complex64},
		{"W^-1", -1, mulW16PowNeg1Complex64},
		{"W^-2", -2, mulW16PowNeg2Complex64},
		{"W^-3", -3, mulW16PowNeg3Complex64},
		{"W^-4", -4, mulPosIComplex64},
		{"W^-6", -6, mulW16PowNeg6Complex64},
	}

	rng := rand.New(rand.NewPCG(0xB16B00B5, 0x5EED))

	for _, tc := range cases {
		for range 64 {
			z := complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
			want := complex(float64(real(z)), float64(imag(z))) * w(tc.exp)
			got := tc.fn(z)

			d := cmplx.Abs(complex(float64(real(got)), float64(imag(got))) - want)
			if d > 1e-5 {
				t.Fatalf("%s: got %v, want %v (|diff| = %g)", tc.name, got, want, d)
			}
		}
	}
}
