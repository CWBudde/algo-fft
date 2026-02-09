//go:build amd64 && asm

package kernels

import (
	"fmt"
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// sse2TestCase64 defines a single SSE2/SSE3 kernel test case for complex64
type sse2TestCase64 struct {
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

// sse2TestCase128 defines a single SSE2 kernel test case for complex128
type sse2TestCase128 struct {
	name          string
	size          int
	radix         int
	tolerance     float64
	forwardSeed   uint64
	inverseSeed   uint64
	roundTripSeed uint64
	forwardKernel func([]complex128, []complex128, []complex128, []complex128) bool
	inverseKernel func([]complex128, []complex128, []complex128, []complex128) bool
}

var sse2TestCases64 = []sse2TestCase64{
	{
		name:      "Size4/Radix4",
		size:      4,
		radix:     4,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size8/Radix2",
		size:      8,
		radix:     2,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size8/Radix4",
		size:      8,
		radix:     4,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size8/Radix8",
		size:      8,
		radix:     8,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size16/Radix2",
		size:      16,
		radix:     2,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size16Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size16Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size16/Radix4",
		size:      16,
		radix:     4,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size16/Radix16",
		size:      16,
		radix:     16,
		tolerance: 1e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size32/Radix2",
		size:      32,
		radix:     2,
		tolerance: 2e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size32/Radix32",
		size:      32,
		radix:     32,
		tolerance: 2e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size32/Radix4Then2",
		size:      32,
		radix:     -24,
		tolerance: 2e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size64/Radix2",
		size:      64,
		radix:     2,
		tolerance: 2e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size64/Radix4",
		size:      64,
		radix:     4,
		tolerance: 2e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size128/Radix2",
		size:      128,
		radix:     2,
		tolerance: 3e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size128/Radix4Then2",
		size:      128,
		radix:     -24,
		tolerance: 3e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size256/Radix4",
		size:      256,
		radix:     4,
		tolerance: 4e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size512/Radix2",
		size:      512,
		radix:     2,
		tolerance: 7e-6,
		forwardKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.ForwardSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex64) bool {
			return amd64.InverseSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch)
		},
	},
}

var sse2TestCases128 = []sse2TestCase128{
	{
		name:      "Size4/Radix4",
		size:      4,
		radix:     4,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size8/Radix2",
		size:      8,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size8/Radix8",
		size:      8,
		radix:     8,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size8/Radix4",
		size:      8,
		radix:     4,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size16/Radix2",
		size:      16,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size16/Radix4",
		size:      16,
		radix:     4,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size32/Radix2",
		size:      32,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size32/Radix4Then2",
		size:      32,
		radix:     -24,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size64/Radix2",
		size:      64,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size64/Radix4",
		size:      64,
		radix:     4,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size128/Radix2",
		size:      128,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size128/Radix4Then2",
		size:      128,
		radix:     -24,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size256/Radix2",
		size:      256,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size256Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size256Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size256/Radix4",
		size:      256,
		radix:     4,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch)
		},
	},
	{
		name:      "Size512/Radix2",
		size:      512,
		radix:     2,
		tolerance: 1e-11,
		forwardKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.ForwardSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
		inverseKernel: func(dst, src, twiddle, scratch []complex128) bool {
			return amd64.InverseSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
		},
	},
}

func TestSSE2Kernels64(t *testing.T) {
	for _, testCase := range sse2TestCases64 {
		t.Run(fmt.Sprintf("%s/Forward", testCase.name), func(t *testing.T) {
			t.Parallel()
			src := randomComplex64(testCase.size, 0x12345678)
			dst := make([]complex64, testCase.size)
			scratch := make([]complex64, testCase.size)
			twiddle := ComputeTwiddleFactors[complex64](testCase.size)

			if !testCase.forwardKernel(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, testCase.tolerance)
		})
		t.Run(fmt.Sprintf("%s/Inverse", testCase.name), func(t *testing.T) {
			t.Parallel()
			src := randomComplex64(testCase.size, 0x87654321)
			dst := make([]complex64, testCase.size)
			scratch := make([]complex64, testCase.size)
			twiddle := ComputeTwiddleFactors[complex64](testCase.size)

			if !testCase.inverseKernel(dst, src, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			want := reference.NaiveIDFT(src)
			assertComplex64Close(t, dst, want, testCase.tolerance)
		})
		t.Run(fmt.Sprintf("%s/RoundTrip", testCase.name), func(t *testing.T) {
			t.Parallel()
			src := randomComplex64(testCase.size, 0xAABBCCDD)
			fwd := make([]complex64, testCase.size)
			inv := make([]complex64, testCase.size)
			scratch := make([]complex64, testCase.size)
			twiddle := ComputeTwiddleFactors[complex64](testCase.size)

			if !testCase.forwardKernel(fwd, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}
			if !testCase.inverseKernel(inv, fwd, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			assertComplex64Close(t, inv, src, testCase.tolerance)
		})
	}
}

func TestSSE2Kernels128(t *testing.T) {
	for _, testCase := range sse2TestCases128 {
		testCase := testCase
		t.Run(fmt.Sprintf("%s/Forward", testCase.name), func(t *testing.T) {
			t.Parallel()
			src := randomComplex128(testCase.size, 0x11223344)
			dst := make([]complex128, testCase.size)
			scratch := make([]complex128, testCase.size)
			twiddle := ComputeTwiddleFactors[complex128](testCase.size)

			if !testCase.forwardKernel(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}

			want := reference.NaiveDFT128(src)
			assertComplex128Close(t, dst, want, testCase.tolerance)
		})
		t.Run(fmt.Sprintf("%s/Inverse", testCase.name), func(t *testing.T) {
			t.Parallel()
			src := randomComplex128(testCase.size, 0x55667788)
			dst := make([]complex128, testCase.size)
			scratch := make([]complex128, testCase.size)
			twiddle := ComputeTwiddleFactors[complex128](testCase.size)

			if !testCase.inverseKernel(dst, src, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			want := reference.NaiveIDFT128(src)
			assertComplex128Close(t, dst, want, testCase.tolerance)
		})
		t.Run(fmt.Sprintf("%s/RoundTrip", testCase.name), func(t *testing.T) {
			t.Parallel()
			src := randomComplex128(testCase.size, 0x99AABBCC)
			fwd := make([]complex128, testCase.size)
			inv := make([]complex128, testCase.size)
			scratch := make([]complex128, testCase.size)
			twiddle := ComputeTwiddleFactors[complex128](testCase.size)

			if !testCase.forwardKernel(fwd, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}
			if !testCase.inverseKernel(inv, fwd, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			assertComplex128Close(t, inv, src, testCase.tolerance)
		})
	}
}
