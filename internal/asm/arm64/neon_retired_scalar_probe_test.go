//go:build arm64 && !purego && fftprobe

package arm64

// Correctness tests and comparison benchmarks for the 12 retired scalar-NEON
// DIT codelets kept behind `-tags fftprobe`. See decl_probe.go for the full
// roster and AGENTS.md "Losing on one machine is not grounds for deletion"
// §2.2 for the disposition this implements ("Measured loss >= 1.5x, or a
// research kernel — keep, unregistered").
//
// Each of these kernels contains zero vector instructions (plain
// FMOVD/FADDD/FMULD scalar arithmetic under a "NEON" name) and measured
// 2.7x-5.6x slower than the pure-Go codelet on an Apple M5 — see
// docs/CODELET_BENCHMARKS.md. The benchmarks below compare each retired
// kernel against the generic NEON kernel that supersedes it in production
// (registered at every affected size via ForwardNEONComplex64Asm /
// ForwardNEONComplex128Asm), so the ratio that justified retiring these
// kernels stays re-measurable on any host instead of becoming folklore.
//
// Take the numbers with:
//
//	go test -tags fftprobe -run '^$' -bench BenchmarkRetiredScalarNEON \
//	  -benchtime=1s ./internal/asm/arm64/

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func computeTwiddle64(n int) []complex64 {
	tw := make([]complex64, n)
	for k := range n {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		sin, cos := math.Sincos(angle)
		tw[k] = complex(float32(cos), float32(sin))
	}

	return tw
}

func checkMaxErr64(t *testing.T, got, want []complex64, tol float64, label string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}

	maxErr := 0.0
	worst := -1

	for i := range got {
		diff := complex128(got[i] - want[i])

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

type retiredScalarProbeCase64 struct {
	name    string
	size    int
	forward func(dst, src, twiddle, scratch []complex64) bool
	inverse func(dst, src, twiddle, scratch []complex64) bool
	tol     float64
}

// Tolerances widen with size, mirroring the pre-retirement table in
// asm_arm64_neon_test.go: complex64 accumulates more rounding error per
// added FFT stage, so a flat bound across 8..256 either fails at 256 or
// hides real regressions at 8.
//
//nolint:gochecknoglobals // probe-only table
var retiredScalarProbeCases64 = []retiredScalarProbeCase64{
	{"Size8_Radix2", 8, ForwardNEONSize8Radix2Complex64Asm, InverseNEONSize8Radix2Complex64Asm, 1e-4},
	{"Size8_Radix4", 8, ForwardNEONSize8Radix4Complex64Asm, InverseNEONSize8Radix4Complex64Asm, 1e-4},
	{"Size16_Radix2", 16, ForwardNEONSize16Radix2Complex64Asm, InverseNEONSize16Radix2Complex64Asm, 1e-4},
	{"Size32_Radix2", 32, ForwardNEONSize32Radix2Complex64Asm, InverseNEONSize32Radix2Complex64Asm, 1e-4},
	{"Size128_Radix2", 128, ForwardNEONSize128Radix2Complex64Asm, InverseNEONSize128Radix2Complex64Asm, 1e-3},
	{"Size256_Radix2", 256, ForwardNEONSize256Radix2Complex64Asm, InverseNEONSize256Radix2Complex64Asm, 5e-3},
}

type retiredScalarProbeCase128 struct {
	name    string
	size    int
	forward func(dst, src, twiddle, scratch []complex128) bool
	inverse func(dst, src, twiddle, scratch []complex128) bool
}

//nolint:gochecknoglobals // probe-only table
var retiredScalarProbeCases128 = []retiredScalarProbeCase128{
	{"Size8_Radix2", 8, ForwardNEONSize8Radix2Complex128Asm, InverseNEONSize8Radix2Complex128Asm},
	{"Size16_Radix2", 16, ForwardNEONSize16Complex128Asm, InverseNEONSize16Complex128Asm},
	{"Size32_Radix2", 32, ForwardNEONSize32Complex128Asm, InverseNEONSize32Complex128Asm},
	{"Size64_Radix2", 64, ForwardNEONSize64Radix2Complex128Asm, InverseNEONSize64Radix2Complex128Asm},
	{"Size128_Radix2", 128, ForwardNEONSize128Radix2Complex128Asm, InverseNEONSize128Radix2Complex128Asm},
	{"Size256_Radix2", 256, ForwardNEONSize256Radix2Complex128Asm, InverseNEONSize256Radix2Complex128Asm},
}

// TestRetiredScalarNEONComplex64 is the guard that matters most here: these
// kernels are unreachable from production and from the registry-driven
// reference sweep, so nothing else in the suite would notice a silent
// correctness regression.
func TestRetiredScalarNEONComplex64(t *testing.T) {
	for _, tc := range retiredScalarProbeCases64 {
		t.Run(tc.name, func(t *testing.T) {
			src := make([]complex64, tc.size)
			for i := range src {
				src[i] = complex(float32(i*3-7), float32(11-i*2))
			}

			twiddle := computeTwiddle64(tc.size)
			scratch := make([]complex64, tc.size)
			dst := make([]complex64, tc.size)

			if !tc.forward(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel returned false")
			}

			ref := reference.NaiveDFT(src)
			checkMaxErr64(t, dst, ref, tc.tol, "forward-vs-reference")

			inv := make([]complex64, tc.size)
			if !tc.inverse(inv, dst, twiddle, scratch) {
				t.Fatal("inverse kernel returned false")
			}

			checkMaxErr64(t, inv, src, tc.tol, "round-trip")
		})
	}
}

// TestRetiredScalarNEONComplex128 is the complex128 twin.
func TestRetiredScalarNEONComplex128(t *testing.T) {
	for _, tc := range retiredScalarProbeCases128 {
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
			checkMaxErr(t, dst, ref, 1e-9, "forward-vs-reference")

			inv := make([]complex128, tc.size)
			if !tc.inverse(inv, dst, twiddle, scratch) {
				t.Fatal("inverse kernel returned false")
			}

			checkMaxErr(t, inv, src, 1e-9, "round-trip")
		})
	}
}

// BenchmarkRetiredScalarNEONComplex64 compares each retired kernel against
// the generic NEON kernel that supersedes it in production dispatch.
func BenchmarkRetiredScalarNEONComplex64(b *testing.B) {
	for _, tc := range retiredScalarProbeCases64 {
		src := make([]complex64, tc.size)
		for i := range src {
			src[i] = complex(float32(i*3-7), float32(11-i*2))
		}

		twiddle := computeTwiddle64(tc.size)
		scratch := make([]complex64, tc.size)
		dst := make([]complex64, tc.size)

		b.Run(tc.name+"/retired", func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				tc.forward(dst, src, twiddle, scratch)
			}
		})

		b.Run(tc.name+"/generic", func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				ForwardNEONComplex64Asm(dst, src, twiddle, scratch)
			}
		})
	}
}

// BenchmarkRetiredScalarNEONComplex128 is the complex128 twin.
func BenchmarkRetiredScalarNEONComplex128(b *testing.B) {
	for _, tc := range retiredScalarProbeCases128 {
		src := make([]complex128, tc.size)
		for i := range src {
			src[i] = complex(float64(i*3%13-7), float64(11-i*5%17))
		}

		twiddle := computeTwiddle128(tc.size)
		scratch := make([]complex128, tc.size)
		dst := make([]complex128, tc.size)

		b.Run(tc.name+"/retired", func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				tc.forward(dst, src, twiddle, scratch)
			}
		})

		b.Run(tc.name+"/generic", func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				ForwardNEONComplex128Asm(dst, src, twiddle, scratch)
			}
		})
	}
}
