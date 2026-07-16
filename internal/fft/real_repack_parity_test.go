package fft

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestRepackInverseComplex64MatchesGeneric verifies that the dispatched
// (possibly SIMD) inverse pre-pass produces the same packed buffer as the
// scalar reference loop.
func TestRepackInverseComplex64MatchesGeneric(t *testing.T) {
	t.Parallel()

	for _, half := range []int{2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 31, 32, 33, 64, 100, 128, 129, 512, 1000, 1024} {
		t.Run(fmt.Sprintf("half=%d", half), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(half)))

			src := make([]complex64, half+1)
			for i := range src {
				src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
			}

			// DC and Nyquist must be real for a valid spectrum.
			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			weight := recombineWeights64(half)

			got := make([]complex64, half)
			RepackInverseComplex64(got, src, weight)

			want := make([]complex64, half)
			x0 := real(src[0])
			xh := real(src[half])
			want[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))
			inverseRepackComplex64Generic(want, src, weight, 1)

			const eps = 1e-5

			for k := range half {
				dr := math.Abs(float64(real(got[k]) - real(want[k])))
				di := math.Abs(float64(imag(got[k]) - imag(want[k])))

				if dr > eps || di > eps {
					t.Errorf("bin %d: got %v, want %v", k, got[k], want[k])
				}
			}
		})
	}
}

// TestRepackInverseComplex128MatchesGeneric verifies the complex128 inverse
// pre-pass against the scalar reference loop.
func TestRepackInverseComplex128MatchesGeneric(t *testing.T) {
	t.Parallel()

	for _, half := range []int{2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 31, 32, 33, 64, 100, 128, 129, 512, 1000, 1024} {
		t.Run(fmt.Sprintf("half=%d", half), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(half) + 8192))

			src := make([]complex128, half+1)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			weight := recombineWeights128(half)

			got := make([]complex128, half)
			RepackInverseComplex128(got, src, weight)

			want := make([]complex128, half)
			x0 := real(src[0])
			xh := real(src[half])
			want[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))
			inverseRepackComplex128Generic(want, src, weight, 1)

			const eps = 1e-14

			for k := range half {
				dr := math.Abs(real(got[k]) - real(want[k]))
				di := math.Abs(imag(got[k]) - imag(want[k]))

				if dr > eps || di > eps {
					t.Errorf("bin %d: got %v, want %v", k, got[k], want[k])
				}
			}
		})
	}
}

func BenchmarkRepackInverseComplex128(b *testing.B) {
	for _, half := range []int{128, 512, 2048, 8192} {
		b.Run(fmt.Sprintf("half=%d", half), func(b *testing.B) {
			src := make([]complex128, half+1)
			for i := range src {
				src[i] = complex(float64(i%17)-8, float64(i%13)-6)
			}

			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			weight := recombineWeights128(half)
			dst := make([]complex128, half)

			b.ReportAllocs()
			b.SetBytes(int64(half * 16))
			b.ResetTimer()

			for range b.N {
				RepackInverseComplex128(dst, src, weight)
			}
		})
	}
}

func BenchmarkRepackInverseComplex128Generic(b *testing.B) {
	for _, half := range []int{128, 512, 2048, 8192} {
		b.Run(fmt.Sprintf("half=%d", half), func(b *testing.B) {
			src := make([]complex128, half+1)
			for i := range src {
				src[i] = complex(float64(i%17)-8, float64(i%13)-6)
			}

			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			weight := recombineWeights128(half)
			dst := make([]complex128, half)

			b.ReportAllocs()
			b.SetBytes(int64(half * 16))
			b.ResetTimer()

			for range b.N {
				x0 := real(src[0])
				xh := real(src[half])
				dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))
				inverseRepackComplex128Generic(dst, src, weight, 1)
			}
		})
	}
}
