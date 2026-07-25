//go:build arm64 && !purego

package arm64

// Validates every size-specific NEON complex128 kernel against the naive DFT,
// including the in-place (dst == src) copy-back path that the registry sweep
// in internal/kernels does not exercise per raw kernel.

import (
	"math"
	"os"
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
		{"Size2048_Radix4Then2", 2048, ForwardNEONSize2048Radix4Then2Complex128Asm, InverseNEONSize2048Radix4Then2Complex128Asm},
		{"Size4096_Radix4", 4096, ForwardNEONSize4096Radix4Complex128Asm, InverseNEONSize4096Radix4Complex128Asm},
		{"Size8192_Radix4Then2", 8192, ForwardNEONSize8192Radix4Then2Complex128Asm, InverseNEONSize8192Radix4Then2Complex128Asm},
		{"Size16384_Radix4", 16384, ForwardNEONSize16384Radix4Complex128Asm, InverseNEONSize16384Radix4Complex128Asm},
		{"Size32768_Radix4Then2", 32768, ForwardNEONSize32768Radix4Then2Complex128Asm, InverseNEONSize32768Radix4Then2Complex128Asm},
	}

	// Size 1024 involves 5 FFT stages with larger-magnitude accumulated sums
	// than the smaller sizes above, so its round-trip error is naturally a
	// bit larger in absolute terms (still ~3e-13 relative, matching the
	// tol=1e-9 relative bound used by codelet_reference_all_test.go for
	// complex128).
	//
	// Sizes 2048 and up add further stages on top of that; each extra stage
	// of summation grows the absolute error while the error relative to the
	// operand magnitude stays flat (~8e-13 at 2048, comfortably within the
	// relative tol=1e-9 bound of the registry tests). Measured max absolute
	// errors: ~1.7e-9 at 2048, ~3.8e-9 at 4096; the larger sizes get
	// tolerances extrapolated on the same growth curve, all with generous
	// headroom above the expected values, instead of widening the shared
	// constant for every smaller kernel. The registry-driven reference tests
	// in internal/kernels remain the authoritative correctness check.
	const tol = 1e-9
	sizeTol := map[int]float64{
		2048:  5e-9,
		4096:  2e-8,
		8192:  5e-8, // passed at this bound (QEMU, 2026-07)
		16384: 1e-7, // passed at this bound (QEMU, 2026-07)
		32768: 1e-6, // measured max ~2.5e-7 (QEMU, 2026-07); relative ~4e-12
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// The naive O(n²) DFT reference at 8192+ takes minutes under QEMU
			// emulation; skip there (mirroring skipNaiveReferenceIfSlow in
			// internal/kernels) and rely on the registry sweep's analytic
			// patterns for those sizes. On real arm64 hardware this runs fully.
			if tc.size >= 8192 && (testing.Short() || os.Getenv("ALGOFFT_QEMU") == "1") {
				t.Skip("skipping naive reference for large size under QEMU/-short")
			}

			tcTol := tol
			if st, ok := sizeTol[tc.size]; ok {
				tcTol = st
			}

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
			checkMaxErr(t, dst, ref, tcTol, "forward-vs-reference")

			inv := make([]complex128, tc.size)
			if !tc.inverse(inv, dst, twiddle, scratch) {
				t.Fatal("inverse kernel returned false")
			}

			checkMaxErr(t, inv, src, tcTol, "round-trip")

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

			checkMaxErr(t, inPlace, ref, tcTol, "in-place-forward")
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
