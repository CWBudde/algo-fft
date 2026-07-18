package kernels

import "math"

// radix7Fwd64/128 hold the full 7x7 DFT coefficient matrix W^(jk mod 7) with
// W = exp(-2*pi*i/7); radix7Inv64/128 hold the conjugate matrix. Row j starts
// at index j*7. Storing both directions avoids a per-term conjugation in the
// inverse butterfly's hot loop.
//
//nolint:gochecknoglobals
var (
	radix7Fwd64  [49]complex64
	radix7Inv64  [49]complex64
	radix7Fwd128 [49]complex128
	radix7Inv128 [49]complex128
)

//nolint:gochecknoinits
func init() {
	for j := range 7 {
		for k := range 7 {
			angle := -2 * math.Pi * float64((j*k)%7) / 7
			re := math.Cos(angle)
			im := math.Sin(angle)
			radix7Fwd128[j*7+k] = complex(re, im)
			radix7Inv128[j*7+k] = complex(re, -im)
			radix7Fwd64[j*7+k] = complex(float32(re), float32(im))
			radix7Inv64[j*7+k] = complex(float32(re), float32(-im))
		}
	}
}

// Butterfly7ForwardComplex64 applies the forward radix-7 DFT butterfly to a
// in place. Inputs must already carry their stage twiddle factors.
func Butterfly7ForwardComplex64(a *[7]complex64) {
	butterfly7Complex64(a, &radix7Fwd64)
}

// Butterfly7InverseComplex64 applies the inverse (conjugate) radix-7 DFT
// butterfly to a in place. No 1/7 scaling is applied.
func Butterfly7InverseComplex64(a *[7]complex64) {
	butterfly7Complex64(a, &radix7Inv64)
}

// Butterfly7ForwardComplex128 applies the forward radix-7 DFT butterfly to a
// in place. Inputs must already carry their stage twiddle factors.
func Butterfly7ForwardComplex128(a *[7]complex128) {
	butterfly7Complex128(a, &radix7Fwd128)
}

// Butterfly7InverseComplex128 applies the inverse (conjugate) radix-7 DFT
// butterfly to a in place. No 1/7 scaling is applied.
func Butterfly7InverseComplex128(a *[7]complex128) {
	butterfly7Complex128(a, &radix7Inv128)
}

func butterfly7Complex64(a *[7]complex64, table *[49]complex64) {
	var y [7]complex64

	sum := a[0]
	for k := 1; k < 7; k++ {
		sum += a[k]
	}

	y[0] = sum

	for j := 1; j < 7; j++ {
		acc := a[0]
		row := table[j*7 : j*7+7]

		for k := 1; k < 7; k++ {
			acc += a[k] * row[k]
		}

		y[j] = acc
	}

	*a = y
}

func butterfly7Complex128(a *[7]complex128, table *[49]complex128) {
	var y [7]complex128

	sum := a[0]
	for k := 1; k < 7; k++ {
		sum += a[k]
	}

	y[0] = sum

	for j := 1; j < 7; j++ {
		acc := a[0]
		row := table[j*7 : j*7+7]

		for k := 1; k < 7; k++ {
			acc += a[k] * row[k]
		}

		y[j] = acc
	}

	*a = y
}
