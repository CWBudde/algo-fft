package algofft

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestZZDSPCorrect(t *testing.T) {
	for _, n := range []int{1000, 2205, 3600, 12000} {
		rng := rand.New(rand.NewSource(42))
		src := make([]complex64, n)
		src128 := make([]complex128, n)
		for i := range src {
			re, im := rng.NormFloat64(), rng.NormFloat64()
			src[i] = complex(float32(re), float32(im))
			src128[i] = complex(re, im)
		}
		want := reference.NaiveDFTWide(src)

		p, err := NewPlan32(n)
		if err != nil {
			t.Fatal(err)
		}
		dst := make([]complex64, n)
		if err := p.Forward(dst, src); err != nil {
			t.Fatal(err)
		}

		p2, err := NewPlan64(n)
		if err != nil {
			t.Fatal(err)
		}
		dst128 := make([]complex128, n)
		if err := p2.Forward(dst128, src128); err != nil {
			t.Fatal(err)
		}

		var maxRel, maxRel128 float64
		for i := range dst {
			m := cmplx.Abs(want[i])
			if m < 1e-9 {
				continue
			}
			maxRel = math.Max(maxRel, cmplx.Abs(complex128(dst[i])-want[i])/m)
			maxRel128 = math.Max(maxRel128, cmplx.Abs(dst128[i]-want[i])/m)
		}
		t.Logf("n=%6d c64 maxRel=%.3g  c128 maxRel=%.3g", n, maxRel, maxRel128)
		if maxRel > 1e-3 || maxRel128 > 1e-3 {
			t.Errorf("n=%d: too large", n)
		}
	}
}
