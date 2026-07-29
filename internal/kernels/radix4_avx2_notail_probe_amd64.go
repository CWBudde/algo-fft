//go:build amd64 && !purego && fftprobe

package kernels

// Temporary measurement probe for the n = 2048 local minimum (PLAN.md §4).
//
// DELETE THIS FILE once the round is written up. It registers codelets that
// compute the WRONG ANSWER on purpose, which is why it sits behind the
// `fftprobe` build tag: no ordinary build, test or benchmark can see it.
//
// What it measures. For n = 2*4^k the radix-4 kernel transforms the two halves
// independently and then runs a separate full pass over the buffer -- the
// radix-2 tail. That extra pass is the suspected cause of the dip at 2048, so
// the first question is what it costs, which bounds what any fusion can win.
//
// How, without touching the assembly. The kernel's only shape knob is r4End.
// Passing n instead of n/2 leaves the executed radix-4 stages bit-identical --
// the stage loop runs while 4m <= r4End, and the next stage after the last one
// would need m = 4^k, i.e. 4m = 4*4^k > 2*4^k = n, so it stops in the same
// place either way -- while the tail's own guard (r4End >= n) then skips it.
// Same code, same permutation, same twiddle planes, one pass fewer.
// TestRadix4AVX2NoTailProbeIsStagesOnly proves that equivalence rather than
// asserting it: it applies the missing radix-2 combine in Go and requires the
// result to equal the real kernel's output exactly.
//
// The probe is registered as a candidate rather than built as a second binary
// on purpose (§2.2): at this effect size on this laptop only a within-binary,
// within-thermal-window comparison is believable. Its priority sits below the
// incumbent's 90 so the sweep's notion of "incumbent" does not change.

import (
	"math/bits"

	"github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// radix4ProbePriority sits below the real kernel's 90 so registry.Lookup keeps
// returning the incumbent and only the candidate sweep sees the probe.
const (
	radix4ProbePriority = 85
	radix4FusedPriority = 86
)

// radix4NoTailShape reports whether n is one of the shapes that has a tail at
// all, i.e. n = 2*4^k within the kernel's supported range.
func radix4NoTailShape(n int) bool {
	limit, ok := radix4AVX2Limit(n)

	return ok && limit != n
}

func forwardRadix4AVX2NoTailComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)
	if !radix4NoTailShape(n) {
		return false
	}

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), n, false, false, 1)
}

func inverseRadix4AVX2NoTailComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)
	if !radix4NoTailShape(n) {
		return false
	}

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), n, true, false, float32(1)/float32(n))
}

func forwardRadix4AVX2NoTailComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if !radix4NoTailShape(n) {
		return false
	}

	return amd64.Radix4Complex128Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), n, false, false, 1)
}

func inverseRadix4AVX2NoTailComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if !radix4NoTailShape(n) {
		return false
	}

	return amd64.Radix4Complex128Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), n, true, false, float64(1)/float64(n))
}

// The fused variant: the same kernel with the radix-2 tail folded into the
// last radix-4 stage. Unlike the no-tail probe this one is CORRECT -- it is a
// candidate under test, not a measurement artefact, and
// TestRadix4AVX2FusedMatchesUnfused requires it to be bit-identical to the
// incumbent. It is registered below the incumbent's priority so the sweep
// ranks the two without changing which one Lookup returns.

func forwardRadix4AVX2FusedComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, false, true, 1)
}

func inverseRadix4AVX2FusedComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex64Asm(
		dst, src, twiddle, scratch, radix4GroupIndices(n), limit, true, true, float32(1)/float32(n),
	)
}

func forwardRadix4AVX2FusedComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex128Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, false, true, 1)
}

func inverseRadix4AVX2FusedComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex128Asm(
		dst, src, twiddle, scratch, radix4GroupIndices(n), limit, true, true, float64(1)/float64(n),
	)
}

// radix4ProbeSizes are the n = 2*4^k sizes at which the real kernel is
// registered in at least one precision.
//
//nolint:gochecknoglobals // probe-only table
var radix4ProbeSizes = []int{128, 512, 2048, 8192, 32768}

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix4ProbeSizes {
		if bits.TrailingZeros(uint(size))%2 == 0 {
			panic("radix4 probe: size is a power of four and has no tail")
		}

		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size:           size,
			Forward:        forwardRadix4AVX2NoTailComplex64,
			Inverse:        inverseRadix4AVX2NoTailComplex64,
			Algorithm:      fftypes.KernelDIT,
			SIMDLevel:      fftypes.SIMDAVX2,
			Signature:      "dit" + itoa(size) + "_radix4_notail_avx2",
			Priority:       radix4ProbePriority,
			KernelType:     fftypes.KernelTypeDIT,
			TwiddleSize:    twiddleSizeRadix4AVX2,
			PrepareTwiddle: prepareTwiddleRadix4AVX2,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size:           size,
			Forward:        forwardRadix4AVX2NoTailComplex128,
			Inverse:        inverseRadix4AVX2NoTailComplex128,
			Algorithm:      fftypes.KernelDIT,
			SIMDLevel:      fftypes.SIMDAVX2,
			Signature:      "dit" + itoa(size) + "_radix4_notail_avx2",
			Priority:       radix4ProbePriority,
			KernelType:     fftypes.KernelTypeDIT,
			TwiddleSize:    twiddleSizeRadix4AVX2Complex128,
			PrepareTwiddle: prepareTwiddleRadix4AVX2Complex128,
		})

		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size:           size,
			Forward:        forwardRadix4AVX2FusedComplex64,
			Inverse:        inverseRadix4AVX2FusedComplex64,
			Algorithm:      fftypes.KernelDIT,
			SIMDLevel:      fftypes.SIMDAVX2,
			Signature:      "dit" + itoa(size) + "_radix4fused_avx2",
			Priority:       radix4FusedPriority,
			KernelType:     fftypes.KernelTypeDIT,
			TwiddleSize:    twiddleSizeRadix4AVX2,
			PrepareTwiddle: prepareTwiddleRadix4AVX2,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size:           size,
			Forward:        forwardRadix4AVX2FusedComplex128,
			Inverse:        inverseRadix4AVX2FusedComplex128,
			Algorithm:      fftypes.KernelDIT,
			SIMDLevel:      fftypes.SIMDAVX2,
			Signature:      "dit" + itoa(size) + "_radix4fused_avx2",
			Priority:       radix4FusedPriority,
			KernelType:     fftypes.KernelTypeDIT,
			TwiddleSize:    twiddleSizeRadix4AVX2Complex128,
			PrepareTwiddle: prepareTwiddleRadix4AVX2Complex128,
		})
	}
}

// itoa avoids pulling strconv into a probe file.
func itoa(v int) string {
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
