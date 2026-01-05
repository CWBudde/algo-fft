//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

func TestButterfly5ForwardAVX2Complex64(t *testing.T) {
	t.Parallel()

	a0 := []complex64{1 + 2i, 3 + 4i}
	a1 := []complex64{-1 + 1i, 2 - 2i}
	a2 := []complex64{0.5 - 1.5i, -3 + 0.25i}
	a3 := []complex64{2 + 0i, -1 - 3i}
	a4 := []complex64{4 - 2i, 0 + 1i}

	y0 := make([]complex64, 2)
	y1 := make([]complex64, 2)
	y2 := make([]complex64, 2)
	y3 := make([]complex64, 2)
	y4 := make([]complex64, 2)

	amd64.Butterfly5ForwardAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4)

	want0 := make([]complex64, 2)
	want1 := make([]complex64, 2)
	want2 := make([]complex64, 2)
	want3 := make([]complex64, 2)
	want4 := make([]complex64, 2)

	for i := range 2 {
		want0[i], want1[i], want2[i], want3[i], want4[i] = butterfly5ForwardComplex64(a0[i], a1[i], a2[i], a3[i], a4[i])
	}

	assertComplex64Close(t, y0, want0, 1e-5)
	assertComplex64Close(t, y1, want1, 1e-5)
	assertComplex64Close(t, y2, want2, 1e-5)
	assertComplex64Close(t, y3, want3, 1e-5)
	assertComplex64Close(t, y4, want4, 1e-5)
}

func TestButterfly5RoundTripAVX2(t *testing.T) {
	t.Parallel()

	orig0 := []complex64{1 + 2i, 3 + 4i}
	orig1 := []complex64{-1 + 3i, 2 - 5i}
	orig2 := []complex64{6 - 1i, -2 + 7i}
	orig3 := []complex64{0.5 + 0.25i, -3 - 2i}
	orig4 := []complex64{4 - 2i, 8 + 5i}

	fwd0 := make([]complex64, 2)
	fwd1 := make([]complex64, 2)
	fwd2 := make([]complex64, 2)
	fwd3 := make([]complex64, 2)
	fwd4 := make([]complex64, 2)

	inv0 := make([]complex64, 2)
	inv1 := make([]complex64, 2)
	inv2 := make([]complex64, 2)
	inv3 := make([]complex64, 2)
	inv4 := make([]complex64, 2)

	amd64.Butterfly5ForwardAVX2Complex64(fwd0, fwd1, fwd2, fwd3, fwd4, orig0, orig1, orig2, orig3, orig4)
	amd64.Butterfly5InverseAVX2Complex64(inv0, inv1, inv2, inv3, inv4, fwd0, fwd1, fwd2, fwd3, fwd4)

	for i := range inv0 {
		inv0[i] /= 5
		inv1[i] /= 5
		inv2[i] /= 5
		inv3[i] /= 5
		inv4[i] /= 5
	}

	assertComplex64Close(t, inv0, orig0, 1e-5)
	assertComplex64Close(t, inv1, orig1, 1e-5)
	assertComplex64Close(t, inv2, orig2, 1e-5)
	assertComplex64Close(t, inv3, orig3, 1e-5)
	assertComplex64Close(t, inv4, orig4, 1e-5)
}

func TestButterfly5InverseAVX2Complex64(t *testing.T) {
	t.Parallel()

	a0 := []complex64{1 + 2i, 3 + 4i}
	a1 := []complex64{-1 + 1i, 2 - 2i}
	a2 := []complex64{0.5 - 1.5i, -3 + 0.25i}
	a3 := []complex64{2 + 0i, -1 - 3i}
	a4 := []complex64{4 - 2i, 0 + 1i}

	y0 := make([]complex64, 2)
	y1 := make([]complex64, 2)
	y2 := make([]complex64, 2)
	y3 := make([]complex64, 2)
	y4 := make([]complex64, 2)

	amd64.Butterfly5InverseAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4)

	want0 := make([]complex64, 2)
	want1 := make([]complex64, 2)
	want2 := make([]complex64, 2)
	want3 := make([]complex64, 2)
	want4 := make([]complex64, 2)

	for i := range 2 {
		want0[i], want1[i], want2[i], want3[i], want4[i] = butterfly5InverseComplex64(a0[i], a1[i], a2[i], a3[i], a4[i])
	}

	assertComplex64Close(t, y0, want0, 1e-5)
	assertComplex64Close(t, y1, want1, 1e-5)
	assertComplex64Close(t, y2, want2, 1e-5)
	assertComplex64Close(t, y3, want3, 1e-5)
	assertComplex64Close(t, y4, want4, 1e-5)
}
