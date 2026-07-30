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
// 45, at n = 512), so the ladder can never be selected by Lookup.
const radix8ProbePriority = 1

// radix8ProbeSizes are the sizes with a pure-Go radix-4 peer to compare
// against. All three ladder shapes are represented: 8^k (64, 512, 4096,
// 32768), 2*8^k (128, 1024, 8192) and 4*8^k (256, 2048, 16384).
//
//nolint:gochecknoglobals // probe-only table
var radix8ProbeSizes = []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix8ProbeSizes {
		// "ladder" distinguishes this from the existing per-size
		// dit512_radix8_generic row, which is a different kernel.
		name := "dit" + itoa(size) + "_radix8ladder_generic"

		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: forwardRadix8Complex64, Inverse: inverseRadix8Complex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature: name, Priority: radix8ProbePriority, KernelType: fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex64,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: forwardRadix8Complex128, Inverse: inverseRadix8Complex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature: name, Priority: radix8ProbePriority, KernelType: fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex128,
		})
	}
}
