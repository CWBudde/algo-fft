//go:build fftprobe

package kernels

// Measurement harness for the size-generic radix-16 ladder. Built only under
// `-tags fftprobe`, so no ordinary build, test or benchmark sees any of it.
//
// What is being tested. Radix-8 earned its assembly by first winning in pure
// Go, where there is no register budget to confound the result -- the ladder
// there is a fair test of passes and butterfly operation count alone. Radix-16
// gets the same test before a single line of assembly is written, because the
// assembly is the expensive part: the AVX2 radix-8 attempt cost ~4-5k lines to
// establish that 16 YMM registers cannot hold 8 live streams plus rotation
// masks, and radix-16 needs 16 live streams. On AVX2 that is hopeless outright;
// on AVX-512 it leaves ~12 scratch ZMM, which is structurally the same losing
// position AVX2 radix-8 was in.
//
// So the prior is a loss, and the honest reasons are worth stating up front:
//
//   - Diminishing passes. Radix-16 makes log2(n)/4 passes against radix-8's
//     log2(n)/3 -- 25% fewer, where radix-8 bought 33% over radix-4. Each step
//     up the ladder buys less than the last.
//   - Growing twiddle cost. 15 planes per stage against radix-8's 7, so the
//     table streamed per stage roughly doubles while the passes saved shrink.
//   - Growing gather cost. Stage 1 is a 16x16 digit-reversed transpose against
//     radix-8's 8x8 -- quadratic in the radix.
//
// A warning sign already in the data: at n = 8192 complex64 the *simplified*
// radix-4 variant `dit8192_radix4_notail_avx2` scored 0.906 against the
// AVX-512 radix-8 ladder's 0.900. A pure simplification came within half a
// percent of an entire new ISA tier. That is what a flattening curve looks
// like.
//
// A loss here is therefore a real answer and closes the radix-16 question for
// the cost of a day rather than a month. A win -- particularly on complex128,
// where the AVX-512 radix-8 sweep was flat at ~0.70 across five sizes, the
// signature of a register-pressure fix rather than a memory effect -- would
// earn an AVX-512 complex128 attempt and nothing wider.
//
// Take the number with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe,purego GOOD=<canary floor> \
//	  taskset -c 0 ./scripts/bench_gated.sh 256 512 1024 2048 4096 8192 16384 32768 65536
//	scripts/bench_gated_analyze.sh benchmarks/gated
//
// The `purego` tag matters. It drops the SIMD tiers so the group's incumbent is
// the pure-Go radix-8 ladder this is being compared against, rather than an
// AVX2 codelet -- which makes the harness's ratio-to-incumbent the answer
// directly. Confirm accepted + rejected equals groups x passes before reading
// any ratio, and take ratios *within* a group: two benchmarks run back to back
// rather than interleaved once put radix-8 6% ahead at n = 512, which the
// interleaved run reversed.
//
// THE RESULT (2026-08-01): radix-16 loses every cell. Do not re-litigate this
// without new information; the numbers below are what the sweep returned, and
// PLAN.md §4 and docs/CODELET_BENCHMARKS.md carry the full record.
//
// 18 groups x 16 passes, 282 accepted + 6 over gate = 288, full accounting,
// GOOD=5216 recalibrated on an idle machine at 46C. Ratios against the pure-Go
// radix-8 ladder, within group:
//
//	n      passes 16:8   c64 fwd  c64 inv   c128 fwd  c128 inv
//	256          2:3       1.158    1.221      1.163     1.225
//	512          3:3       1.138    1.166      1.163     1.158
//	1024         3:4       1.024    1.101      1.018     1.029
//	2048         3:4       1.114    1.115      1.107     1.110
//	4096         3:4       1.139    1.166      1.305     1.356
//	8192         4:5       1.128    1.197      1.298     1.332
//	16384        4:5       1.122    1.138      1.294     1.303
//	32768        4:5       1.126    1.128      1.253     1.278
//
// Radix-16 makes 25-33% fewer passes at every size except 512 and still loses
// everywhere. The pass advantage is real and delivered; the butterfly consumes
// all of it. Radix-8 won 0.87 geomean on this same harness, so the protocol is
// not insensitive -- the radix is. n = 1024 (1.018-1.024 forward) is the
// ceiling, not a lead: it is the size where radix-16's shape is maximally
// favourable, and it still loses the inverse.
//
// n = 65536 is the one uncompared cell -- no radix-8 or radix-4 row is
// registered there, so the probe is its own incumbent and its 1.000 means
// nothing. Registering a radix-8 peer at 65536 is the only way to close it, and
// the trend across 256..32768 is monotone enough that it would not change the
// verdict.
//
// The ladder stays in the tree, behind this tag and out of every production
// build, so the result stays re-measurable rather than becoming folklore.

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// radix16ProbePriority sits below every production generic row, so the probe
// can never be selected by Lookup and only the candidate sweep sees it.
const radix16ProbePriority = 1

// radix16ProbeSizes spans every shape the ladder supports at a size worth
// measuring: 16^k (256, 4096, 65536), 2*16^k (512, 8192), 4*16^k (1024, 16384)
// and 8*16^k (2048, 32768). The three tail radices are each exercised, since a
// tail stage is a full extra pass and could plausibly decide the result on its
// own.
//
//nolint:gochecknoglobals // probe-only table
var radix16ProbeSizes = []int{256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix16ProbeSizes {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: forwardRadix16Complex64, Inverse: inverseRadix16Complex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature:   "dit" + itoa(size) + "_radix16ladder_generic",
			Priority:    radix16ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix16, PrepareTwiddle: prepareTwiddleRadix16Complex64,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: forwardRadix16Complex128, Inverse: inverseRadix16Complex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature:   "dit" + itoa(size) + "_radix16ladder_generic",
			Priority:    radix16ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix16, PrepareTwiddle: prepareTwiddleRadix16Complex128,
		})
	}
}
