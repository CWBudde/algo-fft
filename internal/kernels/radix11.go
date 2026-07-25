package kernels

import (
	"math"

	m "github.com/cwbudde/algo-fft/internal/math"
)

// radix11Fwd64/128 hold the full 11x11 DFT coefficient matrix W^(jk mod 11)
// with W = exp(-2*pi*i/11); radix11Inv64/128 hold the conjugate matrix. Row j
// starts at index j*11. Storing both directions avoids a per-term conjugation
// in the inverse butterfly's hot loop.
//
//nolint:gochecknoglobals
var (
	radix11Fwd64  [121]complex64
	radix11Inv64  [121]complex64
	radix11Fwd128 [121]complex128
	radix11Inv128 [121]complex128
)

//nolint:gochecknoinits
func init() {
	for j := range 11 {
		for k := range 11 {
			angle := -2 * math.Pi * float64((j*k)%11) / 11
			re := math.Cos(angle)
			im := math.Sin(angle)
			radix11Fwd128[j*11+k] = complex(re, im)
			radix11Inv128[j*11+k] = complex(re, -im)
			radix11Fwd64[j*11+k] = complex(float32(re), float32(im))
			radix11Inv64[j*11+k] = complex(float32(re), float32(-im))
		}
	}
}

// Butterfly11ForwardComplex64 applies the forward radix-11 DFT butterfly to a
// in place. Inputs must already carry their stage twiddle factors.
func Butterfly11ForwardComplex64(a *[11]complex64) {
	butterfly11Complex64(a, &radix11Fwd64)
}

// Butterfly11InverseComplex64 applies the inverse (conjugate) radix-11 DFT
// butterfly to a in place. No 1/11 scaling is applied.
func Butterfly11InverseComplex64(a *[11]complex64) {
	butterfly11Complex64(a, &radix11Inv64)
}

func butterfly11Complex64(a *[11]complex64, table *[121]complex64) {
	var y [11]complex64

	sum := a[0]
	for k := 1; k < 11; k++ {
		sum += a[k]
	}

	y[0] = sum

	for j := 1; j < 11; j++ {
		acc := a[0]
		row := table[j*11 : j*11+11]

		for k := 1; k < 11; k++ {
			acc += m.MulComplex64(a[k], row[k])
		}

		y[j] = acc
	}

	*a = y
}
