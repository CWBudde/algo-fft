package kernels

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// naiveDFT8 computes y_m = Σ_j x_j·W_8^{jm} in complex128, with
// sign = -1 for the forward transform and +1 for the (unscaled) inverse.
func naiveDFT8(x [8]complex128, sign float64) [8]complex128 {
	var y [8]complex128

	for m := range 8 {
		var sum complex128
		for j := range 8 {
			angle := sign * 2 * math.Pi * float64(j*m) / 8
			sum += x[j] * cmplx.Exp(complex(0, angle))
		}

		y[m] = sum
	}

	return y
}

func TestButterfly8Complex128(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(42))

	var x [8]complex128
	for i := range x {
		x[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	wantFwd := naiveDFT8(x, -1)
	wantInv := naiveDFT8(x, +1)

	var got [8]complex128

	got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7] = butterfly8ForwardComplex128(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
	for i := range got {
		if cmplx.Abs(got[i]-wantFwd[i]) > 1e-12 {
			t.Errorf("forward mismatch at %d: got %v want %v", i, got[i], wantFwd[i])
		}
	}

	got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7] = butterfly8InverseComplex128(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
	for i := range got {
		if cmplx.Abs(got[i]-wantInv[i]) > 1e-12 {
			t.Errorf("inverse mismatch at %d: got %v want %v", i, got[i], wantInv[i])
		}
	}
}

func TestButterfly8Complex64(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(43))

	var (
		x64  [8]complex64
		x128 [8]complex128
	)

	for i := range x64 {
		re, im := float32(rng.Float64()*2-1), float32(rng.Float64()*2-1)
		x64[i] = complex(re, im)
		x128[i] = complex(float64(re), float64(im))
	}

	wantFwd := naiveDFT8(x128, -1)
	wantInv := naiveDFT8(x128, +1)

	var got [8]complex64

	got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7] = butterfly8ForwardComplex64(x64[0], x64[1], x64[2], x64[3], x64[4], x64[5], x64[6], x64[7])
	for i := range got {
		if cmplx.Abs(complex128(got[i])-wantFwd[i]) > 1e-5 {
			t.Errorf("forward mismatch at %d: got %v want %v", i, got[i], wantFwd[i])
		}
	}

	got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7] = butterfly8InverseComplex64(x64[0], x64[1], x64[2], x64[3], x64[4], x64[5], x64[6], x64[7])
	for i := range got {
		if cmplx.Abs(complex128(got[i])-wantInv[i]) > 1e-5 {
			t.Errorf("inverse mismatch at %d: got %v want %v", i, got[i], wantInv[i])
		}
	}
}
