//go:build amd64 && asm

package amd64

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardAVX2Size256Radix16(t *testing.T) {
	n := 256
	input := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	
	// Generate Twiddles
	twiddle := make([]complex64, n)
	for k := 0; k < n; k++ {
		phase := -2.0 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex64(cmplx.Exp(complex(0, phase)))
	}

	// Generate Random Input
	rnd := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		input[i] = complex(float32(rnd.NormFloat64()), float32(rnd.NormFloat64()))
	}
	
	// Call ASM
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	ok := ForwardAVX2Size256Radix16Complex64Asm(dst, input, twiddle, scratch, indices)
	if !ok {
		t.Fatal("ForwardAVX2Size256Radix16Complex64Asm returned false")
	}

	// Reference
	refInput := make([]complex128, n)
	for i, v := range input {
		refInput[i] = complex128(v)
	}
	refOutput := reference.NaiveDFT128(refInput)

	// Compare
	for i := 0; i < n; i++ {
		got := dst[i]
		want := refOutput[i]
		
		diff := cmplx.Abs(complex128(got) - want)
		if diff > 1e-4 {
			t.Errorf("idx %d: got %v, want %v, diff %g", i, got, want, diff)
			// Print first failure only
			break
		}
	}
}
