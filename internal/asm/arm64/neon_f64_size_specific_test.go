//go:build arm64 && !purego

package arm64

// Validates every size-specific NEON complex128 kernel against the naive DFT,
// including the in-place (dst == src) copy-back path that the registry sweep
// in internal/kernels does not exercise per raw kernel.

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func computeTwiddle128(n int) []complex128 {
	tw := make([]complex128, n)
	for k := range n {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		sin, cos := math.Sincos(angle)
		tw[k] = complex(cos, sin)
	}

	return tw
}

func TestNEONComplex128SizeSpecificKernels(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		forward func(dst, src, twiddle, scratch []complex128) bool
		inverse func(dst, src, twiddle, scratch []complex128) bool
	}{
		{"Size4_Radix4", 4, ForwardNEONSize4Radix4Complex128Asm, InverseNEONSize4Radix4Complex128Asm},
		{"Size8_Radix2", 8, ForwardNEONSize8Radix2Complex128Asm, InverseNEONSize8Radix2Complex128Asm},
		{"Size8_Radix4", 8, ForwardNEONSize8Radix4Complex128Asm, InverseNEONSize8Radix4Complex128Asm},
		{"Size16_Radix2", 16, ForwardNEONSize16Complex128Asm, InverseNEONSize16Complex128Asm},
		{"Size16_Radix4", 16, ForwardNEONSize16Radix4Complex128Asm, InverseNEONSize16Radix4Complex128Asm},
		{"Size32_Radix2", 32, ForwardNEONSize32Complex128Asm, InverseNEONSize32Complex128Asm},
		{"Size32_Radix4Then2", 32, ForwardNEONSize32Radix4Then2Complex128Asm, InverseNEONSize32Radix4Then2Complex128Asm},
		{"Size64_Radix2", 64, ForwardNEONSize64Radix2Complex128Asm, InverseNEONSize64Radix2Complex128Asm},
		{"Size64_Radix4", 64, ForwardNEONSize64Radix4Complex128Asm, InverseNEONSize64Radix4Complex128Asm},
		{"Size128_Radix2", 128, ForwardNEONSize128Radix2Complex128Asm, InverseNEONSize128Radix2Complex128Asm},
		{"Size128_Radix4Then2", 128, ForwardNEONSize128Radix4Then2Complex128Asm, InverseNEONSize128Radix4Then2Complex128Asm},
		{"Size256_Radix2", 256, ForwardNEONSize256Radix2Complex128Asm, InverseNEONSize256Radix2Complex128Asm},
		{"Size256_Radix4", 256, ForwardNEONSize256Radix4Complex128Asm, InverseNEONSize256Radix4Complex128Asm},
		{"Size1024_Radix4", 1024, ForwardNEONSize1024Radix4Complex128Asm, InverseNEONSize1024Radix4Complex128Asm},
	}

	// Size 1024 involves 5 FFT stages with larger-magnitude accumulated sums
	// than the smaller sizes above, so its round-trip error is naturally a
	// bit larger in absolute terms (still ~3e-13 relative, matching the
	// tol=1e-9 relative bound used by codelet_reference_all_test.go for
	// complex128).
	const tol = 1e-9

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := make([]complex128, tc.size)
			for i := range src {
				src[i] = complex(float64(i*3%13-7), float64(11-i*5%17))
			}

			twiddle := computeTwiddle128(tc.size)
			scratch := make([]complex128, tc.size)
			dst := make([]complex128, tc.size)

			if !tc.forward(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel returned false")
			}

			ref := reference.NaiveDFT128(src)
			checkMaxErr(t, dst, ref, tol, "forward-vs-reference")

			inv := make([]complex128, tc.size)
			if !tc.inverse(inv, dst, twiddle, scratch) {
				t.Fatal("inverse kernel returned false")
			}

			checkMaxErr(t, inv, src, tol, "round-trip")

			// In-place (dst == src): the copy-back path where the complex64
			// size-specific kernels once corrupted memory (see PLAN.md P2.2).
			inPlace := make([]complex128, tc.size)
			copy(inPlace, src)

			for i := range scratch {
				scratch[i] = 0
			}

			if !tc.forward(inPlace, inPlace, twiddle, scratch) {
				t.Fatal("in-place forward kernel returned false")
			}

			checkMaxErr(t, inPlace, ref, tol, "in-place-forward")
		})
	}
}

func checkMaxErr(t *testing.T, got, want []complex128, tol float64, label string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}

	maxErr := 0.0
	worst := -1

	for i := range got {
		diff := got[i] - want[i]

		err := math.Hypot(real(diff), imag(diff))
		if err > maxErr {
			maxErr = err
			worst = i
		}
	}

	if maxErr > tol {
		t.Fatalf("%s: max error %e at index %d (got %v want %v) exceeds %e",
			label, maxErr, worst, got[worst], want[worst], tol)
	}
}
