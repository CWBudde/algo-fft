//go:build arm64 && fft_asm && !purego

package fft

import "testing"

func TestNeonComplexMul2Asm(t *testing.T) {
	cases := []struct {
		name string
		a    [2]complex64
		b    [2]complex64
	}{
		{
			name: "RealOnly",
			a:    [2]complex64{complex(1, 0), complex(2, 0)},
			b:    [2]complex64{complex(3, 0), complex(-4, 0)},
		},
		{
			name: "ImagOnly",
			a:    [2]complex64{complex(0, 1), complex(0, -2)},
			b:    [2]complex64{complex(0, 1), complex(0, 3)},
		},
		{
			name: "General",
			a:    [2]complex64{complex(3, 4), complex(-1.5, 2.25)},
			b:    [2]complex64{complex(1, 2), complex(0.5, -0.75)},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			dst := [2]complex64{}
			neonComplexMul2Asm(&dst[0], &tc.a[0], &tc.b[0])

			for i := range dst {
				want := tc.a[i] * tc.b[i]
				if !complexNearEqualArm64(dst[i], want, 1e-5) {
					t.Fatalf("idx=%d got %v want %v", i, dst[i], want)
				}
			}
		})
	}
}

func complexNearEqualArm64(a, b complex64, relTol float32) bool {
	diff := a - b
	diffMag := float32(real(diff)*real(diff) + imag(diff)*imag(diff))
	bMag := float32(real(b)*real(b) + imag(b)*imag(b))

	if bMag > 1e-10 {
		return diffMag <= relTol*relTol*bMag
	}

	return diffMag <= relTol*relTol
}
