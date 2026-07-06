package algofft

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestBisectN40(t *testing.T) {
	n := 40
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatal(err)
	}
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(math.Sin(float64(i)*0.7)+1, math.Cos(float64(i)*0.3))
	}
	want := make([]complex128, n)
	for k := 0; k < n; k++ {
		var sum complex128
		for j := 0; j < n; j++ {
			sum += src[j] * cmplx.Exp(complex(0, -2*math.Pi*float64(k*j)/float64(n)))
		}
		want[k] = sum
	}
	dst := make([]complex128, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatal(err)
	}
	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > 1e-9 {
			t.Fatalf("mismatch at %d: got %v want %v", i, dst[i], want[i])
		}
	}
}
