//go:build amd64 && !purego && fftprobe

package kernels

// Measurement harness for the n = 2*4^k radix-2 tail. Built only under
// `-tags fftprobe`, so no ordinary build, test or benchmark sees any of it.
//
// It exists because the per-size choice recorded in cmd/gencodelets/specs.go --
// which sizes fold the tail into the last radix-4 stage and which keep it as a
// separate pass -- is empirical, and an empirical constant with no way to
// re-derive it rots. Building the candidate sweep with this tag registers, at
// every 2*4^k size:
//
//   - the variant production does NOT use at that size, so the fused/unfused
//     comparison is available in both directions at every size; and
//   - a no-tail probe, which skips the combine entirely.
//
// The no-tail probe computes the WRONG ANSWER on purpose. It is what bounds the
// question: the gap between it and the incumbent is the whole cost of the tail,
// and therefore the most any fusion could ever recover. Measured 9-15% at every
// size, against the 4-6% the fusion actually gets where it wins.
//
// It needs no assembly of its own. The kernel's only shape knob is r4End, and
// passing n instead of n/2 leaves the executed radix-4 stages bit-identical --
// the stage loop runs while 4m <= r4End, and the next stage would need
// 4m = 4*4^k > 2*4^k = n, so it stops in the same place either way -- while the
// tail's own guard (r4End >= n) then skips it.
// TestRadix4AVX2NoTailProbeIsStagesOnly proves that rather than asserting it.
//
// Re-derive the table with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe GOOD=<canary floor> \
//	  taskset -c 0 ./scripts/bench_gated.sh 128 512 2048 8192 32768
//
// All probe priorities sit below the incumbent's 90, so registry.Lookup is
// unaffected and only the candidate sweep sees them.

import (
	"github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

const (
	radix4NoTailPriority = 85
	radix4AltPriority    = 86
)

// radix4TailShape reports whether n has a radix-2 tail at all, i.e. n = 2*4^k
// within the kernel's supported range.
func radix4TailShape(n int) bool {
	limit, ok := radix4AVX2Limit(n)

	return ok && limit != n
}

func forwardRadix4AVX2NoTailComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)
	if !radix4TailShape(n) {
		return false
	}

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), n, false, false, 1)
}

func inverseRadix4AVX2NoTailComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)
	if !radix4TailShape(n) {
		return false
	}

	return amd64.Radix4Complex64Asm(
		dst, src, twiddle, scratch, radix4GroupIndices(n), n, true, false, float32(1)/float32(n),
	)
}

func forwardRadix4AVX2NoTailComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if !radix4TailShape(n) {
		return false
	}

	return amd64.Radix4Complex128Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), n, false, false, 1)
}

func inverseRadix4AVX2NoTailComplex128(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)
	if !radix4TailShape(n) {
		return false
	}

	return amd64.Radix4Complex128Asm(
		dst, src, twiddle, scratch, radix4GroupIndices(n), n, true, false, float64(1)/float64(n),
	)
}

// radix4FusedSizes64 and radix4FusedSizes128 mirror the specs.go rows that
// select the fused variant. The probe registers the OTHER variant at each size,
// so a sweep always has both to compare; keeping these in sync with specs.go is
// what TestRadix4ProbeCoversBothVariants checks.
//
//nolint:gochecknoglobals // probe-only tables
var (
	radix4TailSizes     = []int{128, 512, 2048, 8192, 32768}
	radix4FusedSizes64  = map[int]bool{128: true, 2048: true}
	radix4FusedSizes128 = map[int]bool{128: true}
)

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix4TailSizes {
		fwd64, inv64 := forwardRadix4AVX2FusedComplex64, inverseRadix4AVX2FusedComplex64
		name64 := "dit" + itoa(size) + "_radix4fused_avx2"

		if radix4FusedSizes64[size] {
			fwd64, inv64 = forwardRadix4AVX2Complex64, inverseRadix4AVX2Complex64
			name64 = "dit" + itoa(size) + "_radix4_avx2"
		}

		fwd128, inv128 := forwardRadix4AVX2FusedComplex128, inverseRadix4AVX2FusedComplex128
		name128 := "dit" + itoa(size) + "_radix4fused_avx2"

		if radix4FusedSizes128[size] {
			fwd128, inv128 = forwardRadix4AVX2Complex128, inverseRadix4AVX2Complex128
			name128 = "dit" + itoa(size) + "_radix4_avx2"
		}

		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: fwd64, Inverse: inv64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			Signature: name64, Priority: radix4AltPriority, KernelType: fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix4AVX2, PrepareTwiddle: prepareTwiddleRadix4AVX2,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: fwd128, Inverse: inv128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			Signature: name128, Priority: radix4AltPriority, KernelType: fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix4AVX2Complex128, PrepareTwiddle: prepareTwiddleRadix4AVX2Complex128,
		})

		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size:      size,
			Forward:   forwardRadix4AVX2NoTailComplex64,
			Inverse:   inverseRadix4AVX2NoTailComplex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			Signature: "dit" + itoa(size) + "_radix4_notail_avx2",
			Priority:  radix4NoTailPriority, KernelType: fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix4AVX2, PrepareTwiddle: prepareTwiddleRadix4AVX2,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size:      size,
			Forward:   forwardRadix4AVX2NoTailComplex128,
			Inverse:   inverseRadix4AVX2NoTailComplex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			Signature: "dit" + itoa(size) + "_radix4_notail_avx2",
			Priority:  radix4NoTailPriority, KernelType: fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix4AVX2Complex128, PrepareTwiddle: prepareTwiddleRadix4AVX2Complex128,
		})
	}
}
