//go:build amd64 && !purego && fftprobe

package kernels

// Measurement harness for the size-generic AVX2 radix-8 ladder. Built only
// under `-tags fftprobe`, so no ordinary build, test or benchmark sees any of
// it.
//
// The pure-Go prototype (internal/kernels/radix8_generic.go) is what justified
// writing the assembly: it beat the pure-Go radix-4 ladder by a 0.87 forward
// geomean over 512..32768, and thirteen of its twenty registered cells are now
// production rows. Whether that carries over to 256-bit registers is a
// different question -- the radix-4 AVX2 kernel is far better tuned than the
// pure-Go one it beat, and radix-8 doubles the live streams from four to
// eight, which is exactly what made the radix-4 fused tail lose at n = 2048
// complex128. So the same discipline applies: register the candidate below
// every production row, sweep it against the incumbent inside one group, and
// promote only the cells that measure a win.
//
// Take the number with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe GOOD=<canary floor> \
//	  taskset -c 0 ./scripts/bench_gated.sh 512 1024 2048 4096 8192 16384 32768
//
// radix8AVX2ProbePriority sits below the AVX2 radix-4 rows (90) and below the
// tail probe's 85/86, so registry.Lookup is unaffected and only the candidate
// sweep sees it.

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

const radix8AVX2ProbePriority = 80

// The cells the ladder won on 2026-07-30 now carry real spec rows in
// cmd/gencodelets/specs.go and are not registered here -- a second
// registration would put the same signature in a sweep group twice. What
// remains is where it lost or tied, kept registered so the decision stays
// re-derivable:
//
//	complex64   32, 64 (never measured), 128 (0.984/0.989 -- see below),
//	            4096/8192/16384/32768 (lost, 1.011-1.078 forward: eight streams
//	            4 KiB or more apart)
//	complex128  32, 64 (never measured), 128 (1.026/1.037 -- lost), 256 (1.000
//	            tie), 1024 and 4096 (forward win, inverse loss), 8192 (lost),
//	            16384 (1.012 tie)
//
// n = 128 was swept on 2026-08-01 (GOOD=5216, GATE=1.25, 16 passes, 6 groups,
// 95 accepted + 1 drift = 96, 42 C throughout) and stays unpromoted in both
// precisions. complex128 loses outright at 1.026/1.037. complex64 measures
// 0.984/0.989 against `dit128_radix4fused_avx2`, which is a 1.1-1.6% margin in
// the one group that lost a pass to drift -- below anything this project has
// promoted on, where the bar has been 11-22%.
//
// The same sweep re-derived the complex64 8192/32768 losses at 1.068/1.078 and
// 1.017/1.004, inside the range recorded on 2026-07-30, so the two runs agree
// across five weeks. It also confirmed the existing complex128 32768 radix-8
// spec row from the other direction: radix-4 measures 1.052/1.058 against it.
//
// Do not add a second probe file for these sizes. The lists below already cover
// every 2*4^k cell without a spec row; a sibling probe registering the same
// signature puts the kernel in a sweep group twice, which is what happened on
// 2026-08-01 and cost a full sweep to notice.
//
//nolint:gochecknoglobals // probe-only tables
var (
	radix8AVX2ProbeSizes64  = []int{32, 64, 128, 4096, 8192, 16384, 32768}
	radix8AVX2ProbeSizes128 = []int{32, 64, 128, 256, 1024, 4096, 8192, 16384}
)

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	// "ladder" distinguishes this from the per-size dit512_radix8_avx2 row,
	// which is a different (and XMM-width) kernel.
	for _, size := range radix8AVX2ProbeSizes64 {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: forwardRadix8AVX2Complex64, Inverse: inverseRadix8AVX2Complex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			Signature:   "dit" + itoa(size) + "_radix8ladder_avx2",
			Priority:    radix8AVX2ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex64,
		})
	}

	for _, size := range radix8AVX2ProbeSizes128 {
		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: forwardRadix8AVX2Complex128, Inverse: inverseRadix8AVX2Complex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX2,
			Signature:   "dit" + itoa(size) + "_radix8ladder_avx2",
			Priority:    radix8AVX2ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex128,
		})
	}
}
