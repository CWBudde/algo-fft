//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// avx2TestCase defines a single AVX2 kernel test case
type avx2TestCase struct {
	name          string
	size          int
	radix         int
	tolerance     float64
	forwardSeed   uint64
	inverseSeed   uint64
	roundTripSeed uint64
	forwardKernel func([]complex64, []complex64, []complex64, []complex64) bool
	inverseKernel func([]complex64, []complex64, []complex64, []complex64) bool
}

// avx2TestCases defines all AVX2 kernel test cases
var avx2TestCases = []avx2TestCase{
	{
		name:          "Size4/Radix4",
		size:          4,
		radix:         4,
		tolerance:     1e-6,
		forwardSeed:   0x12345678,
		inverseSeed:   0x87654321,
		roundTripSeed: 0xAABBCCDD,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size8/Radix8",
		size:          8,
		radix:         8,
		tolerance:     1e-6,
		forwardSeed:   0x12345678,
		inverseSeed:   0x87654321,
		roundTripSeed: 0xAABBCCDD,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size16/Radix2",
		size:          16,
		radix:         2,
		tolerance:     1e-6,
		forwardSeed:   0x11223344,
		inverseSeed:   0x55667788,
		roundTripSeed: 0x99AABBCC,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size16/Radix4",
		size:          16,
		radix:         4,
		tolerance:     1e-6,
		forwardSeed:   0x22334455,
		inverseSeed:   0x66778899,
		roundTripSeed: 0xAABBDDEE,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size16/Radix16",
		size:          16,
		radix:         16,
		tolerance:     1e-6,
		forwardSeed:   0x11223344,
		inverseSeed:   0x55667788,
		roundTripSeed: 0x99AABBCC,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size32/Radix2",
		size:          32,
		radix:         2,
		tolerance:     1e-6,
		forwardSeed:   0x33445566,
		inverseSeed:   0x778899AA,
		roundTripSeed: 0xBBCCDDEE,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size32/Radix32",
		size:          32,
		radix:         32,
		tolerance:     1e-6,
		forwardSeed:   0x33445566,
		inverseSeed:   0x778899AA,
		roundTripSeed: 0xBBCCDDEE,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size64/Radix2",
		size:          64,
		radix:         2,
		tolerance:     1e-5,
		forwardSeed:   0x44556677,
		inverseSeed:   0x8899AABB,
		roundTripSeed: 0xCCDDEEFF,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			// Size 64 radix-2 uses bit-reversal, but it's handled internally in the kernel now?
			// Wait, size 64 radix-2 in AVX2 does NOT handle bitrev internally yet?
			// PLAN.md 11.13.2 says Size 64-256 Medium Complex64 Files.
			// "avx2_f32_size64_radix4.s: Internalize bitrev" -> DONE.
			// What about Radix2?
			// codelet_init_avx2.go uses wrapAsmDIT64 with bitrevSize64Radix2.
			// So generic radix-2 DIT needs bitrev.
			// amd64.ForwardAVX2Size64Complex64Asm takes 5 args (via asm_bridge?).
			// Wait, asm_bridge.go defines wrapAsmDIT64 which calls the 5-arg asm func.
			// So the assembly still takes bitrev?
			// Let's check internal/asm/amd64/decl.go.
			// If assembly takes bitrev, I must pass it.
			// But here I'm removing bitrevFunc from test case.
			// So I need to use wrapAsmDIT64 logic inside the lambda?
			return amd64.ForwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch, mathpkg.ComputeBitReversalIndices(64))
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch, mathpkg.ComputeBitReversalIndices(64))
		},
	},
	{
		name:          "Size64/Radix4",
		size:          64,
		radix:         4,
		tolerance:     1e-5,
		forwardSeed:   0x11223344,
		inverseSeed:   0x55667788,
		roundTripSeed: 0x99AABBCC,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size128/Mixed24",
		size:          128,
		radix:         -24,
		tolerance:     1e-5,
		forwardSeed:   0x55667788,
		inverseSeed:   0x99AABBCC,
		roundTripSeed: 0xDDEEFF00,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size256/Radix2",
		size:          256,
		radix:         2,
		tolerance:     1e-5,
		forwardSeed:   0x66778899,
		inverseSeed:   0xAABBCCDD,
		roundTripSeed: 0xEEFF0011,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size256/Radix4",
		size:          256,
		radix:         4,
		tolerance:     1e-5,
		forwardSeed:   0x11223344,
		inverseSeed:   0x55667788,
		roundTripSeed: 0x99AABBCC,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size256/Radix16",
		size:          256,
		radix:         16,
		tolerance:     1e-5,
		forwardSeed:   0x66778899,
		inverseSeed:   0xAABBCCDD,
		roundTripSeed: 0xEEFF0011,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size512/Radix2",
		size:          512,
		radix:         2,
		tolerance:     1e-5,
		forwardSeed:   0x778899AA,
		inverseSeed:   0xBBCCDDEE,
		roundTripSeed: 0xFF001122,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, mathpkg.ComputeBitReversalIndices(512))
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, mathpkg.ComputeBitReversalIndices(512))
		},
	},
	{
		name:          "Size512/Mixed24",
		size:          512,
		radix:         -24,
		tolerance:     1e-5,
		forwardSeed:   0x778899AA,
		inverseSeed:   0xBBCCDDEE,
		roundTripSeed: 0xFF001122,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size1024/Radix4",
		size:          1024,
		radix:         4,
		tolerance:     2e-5,
		forwardSeed:   0x8899AABB,
		inverseSeed:   0xCCDDEEFF,
		roundTripSeed: 0x00112233,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size2048/Mixed24",
		size:          2048,
		radix:         -24,
		tolerance:     2e-5,
		forwardSeed:   0x99AABBCC,
		inverseSeed:   0xDDEEFF00,
		roundTripSeed: 0x11223344,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size4096/Radix4",
		size:          4096,
		radix:         4,
		tolerance:     3e-5,
		forwardSeed:   0xAABBCCDD,
		inverseSeed:   0xEEFF0011,
		roundTripSeed: 0x22334455,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size8192/Mixed24",
		size:          8192,
		radix:         -24,
		tolerance:     5e-5,
		forwardSeed:   0xBBCCDDEE,
		inverseSeed:   0xFF001122,
		roundTripSeed: 0x33445566,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:          "Size16384/Radix4",
		size:          16384,
		radix:         4,
		tolerance:     5e-5,
		forwardSeed:   0xCCDDEEFF,
		inverseSeed:   0x00112233,
		roundTripSeed: 0x44556677,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
}

// TestAVX2KernelsForward tests all AVX2 forward kernels
func TestAVX2KernelsForward(t *testing.T) {
	for _, tc := range avx2TestCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if testing.Short() && tc.size >= 8192 {
				t.Skip("skipping large size in short mode")
			}
			t.Parallel()

			src := randomComplex64(tc.size, tc.forwardSeed)
			dst := make([]complex64, tc.size)
			scratch := make([]complex64, tc.size)
			twiddle := ComputeTwiddleFactors[complex64](tc.size)

			if !tc.forwardKernel(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, tc.tolerance)
		})
	}
}

// TestAVX2KernelsInverse tests all AVX2 inverse kernels
func TestAVX2KernelsInverse(t *testing.T) {
	for _, tc := range avx2TestCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if testing.Short() && tc.size >= 8192 {
				t.Skip("skipping large size in short mode")
			}
			t.Parallel()

			src := randomComplex64(tc.size, tc.inverseSeed)
			fwd := make([]complex64, tc.size)
			dst := make([]complex64, tc.size)
			scratch := make([]complex64, tc.size)
			twiddle := ComputeTwiddleFactors[complex64](tc.size)

			// Generate forward data using reference to ensure valid input
			fwdRef := reference.NaiveDFT(src)
			for i := range fwdRef {
				fwd[i] = complex64(fwdRef[i])
			}

			if !tc.inverseKernel(dst, fwd, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			want := reference.NaiveIDFT(fwdRef)
			assertComplex64Close(t, dst, want, tc.tolerance)
		})
	}
}

// TestAVX2KernelsRoundTrip tests forward-inverse round-trip for all AVX2 kernels
func TestAVX2KernelsRoundTrip(t *testing.T) {
	for _, tc := range avx2TestCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if testing.Short() && tc.size >= 8192 {
				t.Skip("skipping large size in short mode")
			}
			t.Parallel()

			src := randomComplex64(tc.size, tc.roundTripSeed)
			fwd := make([]complex64, tc.size)
			inv := make([]complex64, tc.size)
			scratch := make([]complex64, tc.size)
			twiddle := ComputeTwiddleFactors[complex64](tc.size)

			if !tc.forwardKernel(fwd, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}

			if !tc.inverseKernel(inv, fwd, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			assertComplex64Close(t, inv, src, tc.tolerance)
		})
	}
}