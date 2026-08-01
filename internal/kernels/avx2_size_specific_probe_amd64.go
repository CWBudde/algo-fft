//go:build amd64 && !purego && fftprobe

package kernels

// Measurement harness for the sixteen size-specific AVX2 `.s` files, and for
// the generic AVX2 fallback that would replace them. Built only under
// `-tags fftprobe`, so no ordinary build, test or benchmark sees any of it.
//
// The question it answers is PLAN.md §1.3's first item, which gates the other
// two: *if the generic fallback is within noise of the size-specific kernels,
// the whole question dissolves and ~26,000 lines go with it.*
//
// Why the pair is measurable at all. The size-specific kernels are reached from
// internal/fft/kernels_amd64_size_specific.go, a switch on n that selects by
// KernelStrategy and hands the plan's **raw** DIT twiddle table straight
// through. Every case ends in the same generic fallback -- forwardAVX2Complex64Asm
// / forwardAVX2Complex128Asm -- which consumes the identical raw table. So the
// two arms differ in nothing but the kernel, and both register here with no
// PrepareTwiddle, which is exactly the raw table the switch passes.
//
// Only the FIRST candidate per case is registered. The switch tries several per
// size and returns on the first that accepts, and a size-specific kernel only
// ever rejects a length it was not built for, so candidates two and three are
// already dead code at their own size. Measuring them would answer a question
// production does not ask.
//
// Both arms are RankLevel-demoted to SIMDSSE2 for the reason
// radix8_avx512_probe_amd64.go's header records at length: registry ordering is
// SIMD-level major and Priority orders only within a level, so an arm left at
// SIMDAVX2 would become the incumbent at sizes with no tuned codelet and the
// sweep would report it as 1.000 against itself. SIMDLevel stays SIMDAVX2 so
// execution is still gated on the CPU actually having AVX2.
//
// Take the number with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe \
//	  taskset -c 0 ./scripts/bench_gated.sh 4 8 16 32 64 128 256 512 2048 8192
//	scripts/bench_gated_analyze.sh benchmarks/gated
//
// and read the ratio between the two `sizespec` rows in each group -- not
// either row against the group incumbent, which is a tuned codelet neither arm
// is competing with.

import (
	"sync"

	"github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/registry"
)

const (
	// sizeSpecificProbePriority orders the size-specific arm within the SSE2
	// tier it is demoted into; sizeSpecificFallbackProbePriority puts the
	// fallback arm directly below it so the two are adjacent in the sweep.
	sizeSpecificProbePriority         = 4
	sizeSpecificFallbackProbePriority = 3

	// The generic AVX2 kernels decline below these lengths, so the switch's
	// fallback arm does not exist at complex64 n = 8 or complex128 n = 4: those
	// two cases return false and dispatch walks on to a narrower tier. Registering
	// a declining codelet would fail the reference test with a untouched dst,
	// which is how these two were found.
	sizeSpecificFallbackMinSize64  = 16
	sizeSpecificFallbackMinSize128 = 8
)

// probeBitrev memoises the plain bit-reversal table the complex128 generic
// fallback is handed. internal/fft keeps the same table behind
// cachedBitReversalIndices; recomputing it per call would time the table build
// rather than the kernel.
//
//nolint:gochecknoglobals // probe-only cache
var (
	probeBitrevMu sync.Mutex
	probeBitrev   = map[int][]int{}
)

func probeBitrevIndices(n int) []int {
	probeBitrevMu.Lock()
	defer probeBitrevMu.Unlock()

	if idx, ok := probeBitrev[n]; ok {
		return idx
	}

	idx := m.ComputeBitReversalIndices(n)
	probeBitrev[n] = idx

	return idx
}

// The four functions below mirror internal/fft's generic AVX2 fallbacks
// exactly, including the complex64 radix-4 preamble and the complex128 lack of
// one. They are duplicated rather than exported from internal/fft because that
// package imports this one.

func probeFallbackForwardComplex64(dst, src, twiddle, scratch []complex64) bool {
	if len(src) >= 64 && m.IsPowerOf4(len(src)) {
		if amd64.ForwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}

	if len(src) >= 64 && m.IsPowerOf2(len(src)) {
		if amd64.ForwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}

	return amd64.ForwardAVX2Complex64Asm(dst, src, twiddle, scratch, nil)
}

func probeFallbackInverseComplex64(dst, src, twiddle, scratch []complex64) bool {
	if len(src) >= 64 && m.IsPowerOf4(len(src)) {
		if amd64.InverseAVX2Complex64Radix4Asm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}

	if len(src) >= 64 && m.IsPowerOf2(len(src)) {
		if amd64.InverseAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch, nil) {
			return true
		}
	}

	return amd64.InverseAVX2Complex64Asm(dst, src, twiddle, scratch, nil)
}

func probeFallbackForwardComplex128(dst, src, twiddle, scratch []complex128) bool {
	return amd64.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, probeBitrevIndices(len(src)))
}

func probeFallbackInverseComplex128(dst, src, twiddle, scratch []complex128) bool {
	return amd64.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, probeBitrevIndices(len(src)))
}

// sizeSpecificProbe64 and sizeSpecificProbe128 list, per size, the kernel the
// switch in internal/fft/kernels_amd64_size_specific.go actually reaches --
// the first candidate of that case, under the name of the `.s` file that
// defines it.
//
//nolint:gochecknoglobals // probe-only tables
var sizeSpecificProbe64 = []struct {
	size    int
	name    string
	forward func(dst, src, twiddle, scratch []complex64) bool
	inverse func(dst, src, twiddle, scratch []complex64) bool
}{
	{8, "radix4", amd64.ForwardAVX2Size8Radix4Complex64Asm, amd64.InverseAVX2Size8Radix4Complex64Asm},
	{16, "radix4", amd64.ForwardAVX2Size16Radix4Complex64Asm, amd64.InverseAVX2Size16Radix4Complex64Asm},
	{32, "radix32", amd64.ForwardAVX2Size32Radix32Complex64Asm, amd64.InverseAVX2Size32Radix32Complex64Asm},
	{64, "radix4", amd64.ForwardAVX2Size64Radix4Complex64Asm, amd64.InverseAVX2Size64Radix4Complex64Asm},
	{
		128, "radix4then2",
		amd64.ForwardAVX2Size128Radix4Then2Complex64Asm, amd64.InverseAVX2Size128Radix4Then2Complex64Asm,
	},
	{256, "radix4", amd64.ForwardAVX2Size256Radix4Complex64Asm, amd64.InverseAVX2Size256Radix4Complex64Asm},
	{
		512, "radix4then2",
		amd64.ForwardAVX2Size512Radix4Then2Complex64Asm, amd64.InverseAVX2Size512Radix4Then2Complex64Asm,
	},
	{
		2048, "radix4then2",
		amd64.ForwardAVX2Size2048Radix4Then2Complex64Asm, amd64.InverseAVX2Size2048Radix4Then2Complex64Asm,
	},
	{
		8192, "radix4then2",
		amd64.ForwardAVX2Size8192Radix4Then2Complex64Asm, amd64.InverseAVX2Size8192Radix4Then2Complex64Asm,
	},
}

//nolint:gochecknoglobals // probe-only tables
var sizeSpecificProbe128 = []struct {
	size    int
	name    string
	forward func(dst, src, twiddle, scratch []complex128) bool
	inverse func(dst, src, twiddle, scratch []complex128) bool
}{
	{4, "radix4", amd64.ForwardAVX2Size4Radix4Complex128Asm, amd64.InverseAVX2Size4Radix4Complex128Asm},
	{8, "radix4", amd64.ForwardAVX2Size8Radix4Complex128Asm, amd64.InverseAVX2Size8Radix4Complex128Asm},
	{16, "radix4", amd64.ForwardAVX2Size16Radix4Complex128Asm, amd64.InverseAVX2Size16Radix4Complex128Asm},
	{
		32, "radix4then2",
		amd64.ForwardAVX2Size32Radix4Then2Complex128Asm, amd64.InverseAVX2Size32Radix4Then2Complex128Asm,
	},
	{64, "radix4", amd64.ForwardAVX2Size64Radix4Complex128Asm, amd64.InverseAVX2Size64Radix4Complex128Asm},
	{
		512, "radix4then2",
		amd64.ForwardAVX2Size512Radix4Then2Complex128Asm, amd64.InverseAVX2Size512Radix4Then2Complex128Asm,
	},
}

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, p := range sizeSpecificProbe64 {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: p.size, Forward: p.forward, Inverse: p.inverse,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			RankLevel:  fftypes.SIMDSSE2,
			Signature:  "sizespec" + itoa(p.size) + "_" + p.name + "_avx2",
			Priority:   sizeSpecificProbePriority,
			KernelType: fftypes.KernelTypeDIT,
		})

		if p.size < sizeSpecificFallbackMinSize64 {
			continue
		}

		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: p.size, Forward: probeFallbackForwardComplex64, Inverse: probeFallbackInverseComplex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			RankLevel:  fftypes.SIMDSSE2,
			Signature:  "sizespec" + itoa(p.size) + "_fallback_avx2",
			Priority:   sizeSpecificFallbackProbePriority,
			KernelType: fftypes.KernelTypeDIT,
		})
	}

	for _, p := range sizeSpecificProbe128 {
		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: p.size, Forward: p.forward, Inverse: p.inverse,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			RankLevel:  fftypes.SIMDSSE2,
			Signature:  "sizespec" + itoa(p.size) + "_" + p.name + "_avx2",
			Priority:   sizeSpecificProbePriority,
			KernelType: fftypes.KernelTypeDIT,
		})

		if p.size < sizeSpecificFallbackMinSize128 {
			continue
		}

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: p.size, Forward: probeFallbackForwardComplex128, Inverse: probeFallbackInverseComplex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			RankLevel:  fftypes.SIMDSSE2,
			Signature:  "sizespec" + itoa(p.size) + "_fallback_avx2",
			Priority:   sizeSpecificFallbackProbePriority,
			KernelType: fftypes.KernelTypeDIT,
		})
	}
}
