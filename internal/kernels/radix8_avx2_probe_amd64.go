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
//	complex64   32, 64, 128 (never measured), 4096/8192/16384/32768 (lost,
//	            1.011-1.078 forward: eight streams 4 KiB or more apart)
//	complex128  32, 64, 128 (never measured), 256 (1.000 tie), 1024 and 4096
//	            (forward win, inverse loss), 8192 (lost), 16384 (1.012 tie)
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
