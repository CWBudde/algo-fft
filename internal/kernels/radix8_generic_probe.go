//go:build fftprobe

package kernels

// Measurement harness for the size-generic radix-8 ladder. Built only under
// `-tags fftprobe`, so no ordinary build, test or benchmark sees any of it.
//
// It exists to answer one question that this repository has never actually
// measured: does radix-8 beat radix-4 when the butterfly is written correctly?
// Every earlier radix-8 kernel here lost while making fewer passes over the
// buffer, but each of them threw the advantage away somewhere else -- the AVX2
// size-512 kernel is XMM-width throughout, and the pure-Go size-512 codelet
// spends full complex multiplies on the fixed eighth-roots. Neither result
// tests the algorithm.
//
// The ladder is registered at every size where a pure-Go radix-4 peer exists,
// at a priority below every production row, so registry.Lookup is unaffected
// and only the candidate sweep sees it.
//
// Take the number with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe,purego GOOD=<canary floor> \
//	  taskset -c 0 ./scripts/bench_gated.sh 512 1024 2048 4096 8192 16384 32768
//
// The `purego` tag matters: it drops the SIMD tiers, so the group's incumbent
// is the pure-Go radix-4 kernel this is being compared against rather than an
// AVX2 codelet, and the harness's ratio-to-incumbent is the answer directly.
// Expected shape of a win, from the pass ratios (radix-8 : radix-4):
// 3:5 at 512, 4:5 at 1024, 4:6 at 2048 and 4096, 5:7 at 8192, 5:8 at 32768.

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// radix8ProbePriority sits below every production generic row (the highest is
// 50, the promoted ladder itself), so the probe can never be selected by
// Lookup.
const radix8ProbePriority = 1

// Every size the ladder won now carries a real spec row in
// cmd/gencodelets/specs.go and is not registered here -- a second registration
// would put the same signature in a sweep group twice. What remains are the
// sizes it lost, kept registered so they stay measurable:
//
//	64, 128     forward 1.05-1.11 (both precisions)
//	1024 c128   forward 1.097
//	32768       forward 1.06/1.08 (both precisions)
//
// The four sizes that tied on forward -- 256 and 2048 complex64, 256 and 512
// complex128 -- were re-measured on 2026-07-30 after the 1/n scaling sweep had
// removed the incumbents' trailing complex-multiply pass, so that both sides of
// the comparison were current. All four still won on inverse by 11-22% while
// tying on forward, and all four are now promoted. (An intermediate partial
// re-sweep put radix-4-then-2 back in front at 512 complex128; it was taken on
// a contended machine and the gated run contradicts it.)
//
// n = 32768 is the interesting loss: it has the best pass ratio of the ladder
// (5 against 8) and still loses forward. Its last radix-8 stage holds eight
// streams 4096 elements apart -- 32 KiB at complex64, 64 KiB at complex128 --
// so all eight land on the same L1 sets. That is the same collision the
// radix-4 fused tail hits at n = 2048 complex128 with a 4 KiB stride; see
// forwardRadix4AVX2FusedComplex64. Blocking the stage would test it.
//
//nolint:gochecknoglobals // probe-only tables
var (
	radix8ProbeSizes    = []int{64, 128, 32768}
	radix8ProbeSizes128 = []int{64, 128, 1024, 32768}
)

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	// "ladder" distinguishes this from the existing per-size
	// dit512_radix8_generic row, which is a different kernel.
	for _, size := range radix8ProbeSizes {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: forwardRadix8Complex64, Inverse: inverseRadix8Complex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature:   "dit" + itoa(size) + "_radix8ladder_generic",
			Priority:    radix8ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex64,
		})
	}

	for _, size := range radix8ProbeSizes128 {
		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: forwardRadix8Complex128, Inverse: inverseRadix8Complex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature:   "dit" + itoa(size) + "_radix8ladder_generic",
			Priority:    radix8ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex128,
		})
	}
}
