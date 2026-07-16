package fft

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func recombineWeights64(half int) []complex64 {
	n := half * 2

	weight := make([]complex64, half+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)
		weight[k] = complex(float32(0.5*(1+math.Sin(theta))), float32(0.5*math.Cos(theta)))
	}

	return weight
}

func recombineWeights128(half int) []complex128 {
	n := half * 2

	weight := make([]complex128, half+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)
		weight[k] = complex(0.5*(1+math.Sin(theta)), 0.5*math.Cos(theta))
	}

	return weight
}

// TestRecombineForwardComplex64MatchesGeneric verifies that the dispatched
// (possibly SIMD) path produces the same bins as the scalar reference loop.
// The sizes cover full vector blocks, scalar tails, and sub-vector inputs.
func TestRecombineForwardComplex64MatchesGeneric(t *testing.T) {
	t.Parallel()

	for _, half := range []int{2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 31, 32, 33, 64, 100, 128, 129, 512, 1000, 1024} {
		t.Run(fmt.Sprintf("half=%d", half), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(half)))

			src := make([]complex64, half)
			for i := range src {
				src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
			}

			weight := recombineWeights64(half)

			got := make([]complex64, half+1)
			RecombineForwardComplex64(got, src, weight)

			want := make([]complex64, half+1)
			recombineForwardComplex64Generic(want, src, weight, 1)

			// FMA in the SIMD path rounds once where the scalar loop rounds
			// twice, so allow a couple of float32 ulps of slack.
			const eps = 1e-5

			for k := 1; k < half; k++ {
				dr := math.Abs(float64(real(got[k]) - real(want[k])))
				di := math.Abs(float64(imag(got[k]) - imag(want[k])))

				if dr > eps || di > eps {
					t.Errorf("bin %d: got %v, want %v", k, got[k], want[k])
				}
			}

			// DC and Nyquist bins must be left untouched.
			if got[0] != 0 || got[half] != 0 {
				t.Errorf("DC/Nyquist bins modified: got[0]=%v got[half]=%v", got[0], got[half])
			}
		})
	}
}

// TestRecombineForwardComplex128MatchesGeneric verifies the complex128 path
// against the scalar reference loop.
func TestRecombineForwardComplex128MatchesGeneric(t *testing.T) {
	t.Parallel()

	for _, half := range []int{2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 31, 32, 33, 64, 100, 128, 129, 512, 1000, 1024} {
		t.Run(fmt.Sprintf("half=%d", half), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(half) + 4096))

			src := make([]complex128, half)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			weight := recombineWeights128(half)

			got := make([]complex128, half+1)
			RecombineForwardComplex128(got, src, weight)

			want := make([]complex128, half+1)
			recombineForwardComplex128Generic(want, src, weight, 1)

			const eps = 1e-14

			for k := 1; k < half; k++ {
				dr := math.Abs(real(got[k]) - real(want[k]))
				di := math.Abs(imag(got[k]) - imag(want[k]))

				if dr > eps || di > eps {
					t.Errorf("bin %d: got %v, want %v", k, got[k], want[k])
				}
			}

			if got[0] != 0 || got[half] != 0 {
				t.Errorf("DC/Nyquist bins modified: got[0]=%v got[half]=%v", got[0], got[half])
			}
		})
	}
}

// TestRecombineForwardTinyInputs ensures sub-minimal sizes are a no-op and do
// not panic.
func TestRecombineForwardTinyInputs(t *testing.T) {
	t.Parallel()

	RecombineForwardComplex64(nil, nil, nil)
	RecombineForwardComplex128(nil, nil, nil)

	dst64 := make([]complex64, 2)
	RecombineForwardComplex64(dst64, make([]complex64, 1), make([]complex64, 2))

	dst128 := make([]complex128, 2)
	RecombineForwardComplex128(dst128, make([]complex128, 1), make([]complex128, 2))

	for i := range dst64 {
		if dst64[i] != 0 || dst128[i] != 0 {
			t.Errorf("tiny input modified dst at %d", i)
		}
	}
}

func BenchmarkRecombineForwardComplex64(b *testing.B) {
	for _, half := range []int{128, 512, 2048, 8192} {
		b.Run(fmt.Sprintf("half=%d", half), func(b *testing.B) {
			src := make([]complex64, half)
			for i := range src {
				src[i] = complex(float32(i%17)-8, float32(i%13)-6)
			}

			weight := recombineWeights64(half)
			dst := make([]complex64, half+1)

			b.ReportAllocs()
			b.SetBytes(int64(half * 8))
			b.ResetTimer()

			for range b.N {
				RecombineForwardComplex64(dst, src, weight)
			}
		})
	}
}

func BenchmarkRecombineForwardComplex128(b *testing.B) {
	for _, half := range []int{128, 512, 2048, 8192} {
		b.Run(fmt.Sprintf("half=%d", half), func(b *testing.B) {
			src := make([]complex128, half)
			for i := range src {
				src[i] = complex(float64(i%17)-8, float64(i%13)-6)
			}

			weight := recombineWeights128(half)
			dst := make([]complex128, half+1)

			b.ReportAllocs()
			b.SetBytes(int64(half * 16))
			b.ResetTimer()

			for range b.N {
				RecombineForwardComplex128(dst, src, weight)
			}
		})
	}
}

func BenchmarkRecombineForwardComplex64Generic(b *testing.B) {
	for _, half := range []int{128, 512, 2048, 8192} {
		b.Run(fmt.Sprintf("half=%d", half), func(b *testing.B) {
			src := make([]complex64, half)
			for i := range src {
				src[i] = complex(float32(i%17)-8, float32(i%13)-6)
			}

			weight := recombineWeights64(half)
			dst := make([]complex64, half+1)

			b.ReportAllocs()
			b.SetBytes(int64(half * 8))
			b.ResetTimer()

			for range b.N {
				recombineForwardComplex64Generic(dst, src, weight, 1)
			}
		})
	}
}

func BenchmarkRecombineForwardComplex128Generic(b *testing.B) {
	for _, half := range []int{128, 512, 2048, 8192} {
		b.Run(fmt.Sprintf("half=%d", half), func(b *testing.B) {
			src := make([]complex128, half)
			for i := range src {
				src[i] = complex(float64(i%17)-8, float64(i%13)-6)
			}

			weight := recombineWeights128(half)
			dst := make([]complex128, half+1)

			b.ReportAllocs()
			b.SetBytes(int64(half * 16))
			b.ResetTimer()

			for range b.N {
				recombineForwardComplex128Generic(dst, src, weight, 1)
			}
		})
	}
}
