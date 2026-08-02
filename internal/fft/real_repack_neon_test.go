//go:build arm64 && !purego

package fft

import (
	"math"
	"testing"

	arm64 "github.com/cwbudde/algo-fft/internal/asm/arm64"
)

func TestInverseRepackComplex64NEON(t *testing.T) {
	cases := []int{8, 9, 16, 17, 32, 33}
	for _, half := range cases {
		t.Run("HalfSize_"+itoaRepackNEON(half), func(t *testing.T) {
			runInverseRepackComplex64NEONCase(t, half, func(k int) complex64 {
				n := half * 2
				theta := 2 * math.Pi * float64(k) / float64(n)
				re := 0.5 * (1 + math.Sin(theta))
				im := 0.5 * math.Cos(theta)
				return complex(float32(re), float32(im))
			})
		})
	}
}

func TestInverseRepackComplex64NEON_ZeroWeight(t *testing.T) {
	half := 16
	runInverseRepackComplex64NEONCase(t, half, func(int) complex64 { return 0 })
}

func TestInverseRepackComplex64NEON_PureRealWeight(t *testing.T) {
	runInverseRepackComplex64NEONCase(t, 16, func(int) complex64 { return complex(0.25, 0) })
}

func TestInverseRepackComplex64NEON_PureImagWeight(t *testing.T) {
	runInverseRepackComplex64NEONCase(t, 16, func(int) complex64 { return complex(0, 0.25) })
}

func runInverseRepackComplex64NEONCase(t *testing.T, half int, weightFn func(k int) complex64) {
	t.Helper()

	src := make([]complex64, half+1)
	for i := range src {
		src[i] = complex(float32(i%7-3), float32((i*3)%5-2))
	}
	src[0] = complex(real(src[0]), 0)
	src[half] = complex(real(src[half]), 0)

	weight := make([]complex64, half+1)
	for k := 0; k <= half; k++ {
		weight[k] = weightFn(k)
	}

	dstGeneric := make([]complex64, half)
	dstSIMD := make([]complex64, half)

	x0 := real(src[0])
	xh := real(src[half])
	dstGeneric[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))
	dstSIMD[0] = dstGeneric[0]

	inverseRepackComplex64Generic(dstGeneric, src, weight, 1)

	// Same block contract as inverseRepackComplex64SIMD in
	// real_repack_arm64.go: count is a multiple of 2, <= (half-1)/2.
	count := (half - 1) / 2 / 2 * 2
	start := 1
	if count >= 2 {
		arm64.InverseRepackComplex64NEONAsm(dstSIMD, src, weight, count)
		start = count + 1
	}
	inverseRepackComplex64Generic(dstSIMD, src, weight, start)

	const eps = 1e-4
	for i := range dstGeneric {
		gr := real(dstGeneric[i])
		gi := imag(dstGeneric[i])
		sr := real(dstSIMD[i])
		si := imag(dstSIMD[i])
		if math.Abs(float64(gr-sr)) > eps || math.Abs(float64(gi-si)) > eps {
			t.Fatalf("half=%d idx=%d got=%v want=%v", half, i, dstSIMD[i], dstGeneric[i])
		}
	}
}

func TestInverseRepackComplex128NEON(t *testing.T) {
	cases := []int{8, 9, 16, 17, 32, 33}
	for _, half := range cases {
		t.Run("HalfSize_"+itoaRepackNEON(half), func(t *testing.T) {
			runInverseRepackComplex128NEONCase(t, half, func(k int) complex128 {
				n := half * 2
				theta := 2 * math.Pi * float64(k) / float64(n)
				re := 0.5 * (1 + math.Sin(theta))
				im := 0.5 * math.Cos(theta)
				return complex(re, im)
			})
		})
	}
}

func TestInverseRepackComplex128NEON_ZeroWeight(t *testing.T) {
	half := 16
	runInverseRepackComplex128NEONCase(t, half, func(int) complex128 { return 0 })
}

func TestInverseRepackComplex128NEON_PureRealWeight(t *testing.T) {
	runInverseRepackComplex128NEONCase(t, 16, func(int) complex128 { return complex(0.25, 0) })
}

func TestInverseRepackComplex128NEON_PureImagWeight(t *testing.T) {
	runInverseRepackComplex128NEONCase(t, 16, func(int) complex128 { return complex(0, 0.25) })
}

func runInverseRepackComplex128NEONCase(t *testing.T, half int, weightFn func(k int) complex128) {
	t.Helper()

	src := make([]complex128, half+1)
	for i := range src {
		src[i] = complex(float64(i%7-3), float64((i*3)%5-2))
	}
	src[0] = complex(real(src[0]), 0)
	src[half] = complex(real(src[half]), 0)

	weight := make([]complex128, half+1)
	for k := 0; k <= half; k++ {
		weight[k] = weightFn(k)
	}

	dstGeneric := make([]complex128, half)
	dstSIMD := make([]complex128, half)

	x0 := real(src[0])
	xh := real(src[half])
	dstGeneric[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))
	dstSIMD[0] = dstGeneric[0]

	inverseRepackComplex128Generic(dstGeneric, src, weight, 1)

	// Same block contract as inverseRepackComplex128SIMD in
	// real_repack_arm64.go: count is a multiple of 2, <= (half-1)/2.
	count := (half - 1) / 2 / 2 * 2
	start := 1
	if count >= 2 {
		arm64.InverseRepackComplex128NEONAsm(dstSIMD, src, weight, count)
		start = count + 1
	}
	inverseRepackComplex128Generic(dstSIMD, src, weight, start)

	const eps = 1e-12
	for i := range dstGeneric {
		gr := real(dstGeneric[i])
		gi := imag(dstGeneric[i])
		sr := real(dstSIMD[i])
		si := imag(dstSIMD[i])
		if math.Abs(gr-sr) > eps || math.Abs(gi-si) > eps {
			t.Fatalf("half=%d idx=%d got=%v want=%v", half, i, dstSIMD[i], dstGeneric[i])
		}
	}
}

func itoaRepackNEON(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
