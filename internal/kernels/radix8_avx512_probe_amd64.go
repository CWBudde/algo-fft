//go:build amd64 && !purego && fftprobe

package kernels

// Measurement harness for the size-generic AVX-512 radix-8 ladder. Built only
// under `-tags fftprobe`, so no ordinary build, test or benchmark sees any of
// it.
//
// What is being tested. The 256-bit radix-8 ladder retires a third fewer
// passes over the buffer than radix-4 but costs 1.24-1.56x per pass at
// n = 512..2048 -- sizes that fit entirely in L1, so the penalty is not a
// memory effect and not a spill (the frame is $0 and the kernel touches no
// stack slot). It is the register budget: eight live streams plus two rotation
// masks and a sqrt(2)/2 broadcast leave five scratch YMM of sixteen, which is
// exactly one butterfly's worth. That forces the twiddle planes to be
// re-broadcast from memory every iteration and leaves nothing spare to keep a
// second butterfly in flight across a radix-8's three-level dependency chain,
// where radix-4's chain is two deep and has registers left over.
//
// Thirty-two ZMM leave twenty-one scratch. If the diagnosis is right, that is
// the whole gap, and the ladder should land 1.4-1.7x ahead of the radix-4
// incumbent at n = 8192. If it is wrong, the ladder lands near parity and the
// right AVX-512 target is a wider radix-4 instead -- which is exactly why this
// registers as a probe rather than as a production row.
//
// Take the number on an idle AVX-512 host with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe \
//	  taskset -c 0 ./scripts/bench_gated.sh 64 128 256 512 1024 2048 4096 8192 16384 32768
//	scripts/bench_gated_analyze.sh benchmarks/gated
//
// Keeping the probe out of the incumbent slot takes RankLevel, not Priority.
// Registry ordering is SIMD-level major and Priority only orders within a
// level, so an entry registered at SIMDAVX512 outranks every AVX2 row no matter
// how low its Priority is -- and at 512, 2048 and 32768 there is no production
// AVX-512 row at all, while the radix-2 rows at 1024/4096/8192/16384 are
// themselves RankLevel-demoted to SIMDAVX2. A probe registered in the AVX-512
// tier would therefore *be* the incumbent at ten of the eleven sizes here, and
// the sweep would dutifully report it as 1.000 against itself. The first run of
// this sweep did exactly that.
//
// RankLevel: SIMDSSE2 sorts the probe below every AVX2 and AVX-512 row while
// SIMDLevel: SIMDAVX512 still gates execution, which is the sanctioned
// direction for RankLevel -- demote a wide-ISA codelet, never promote a narrow
// one. It is SSE2 rather than SIMDNone because SIMDNone is the zero value and
// registry.rank() reads it as "unset", falling straight back to SIMDLevel: the
// demotion would silently do nothing. SSE2 is the lowest level that is not the
// zero value, and the size-generic AVX2 radix-4 rows cover every size swept
// here, so the incumbent is always a real codelet. Priority then only orders
// the probe within the SSE2 tier, and is set below those rows too.

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

const radix8AVX512ProbePriority = 5

// Every size each kernel accepts. Nothing is held back: no cell of this
// ladder has been measured yet, and the sizes below 512 are where the
// per-size AVX-512 codelets are strongest, so they are the ones most likely to
// say no.
//
//nolint:gochecknoglobals // probe-only tables
var (
	radix8AVX512ProbeSizes64  = []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
	radix8AVX512ProbeSizes128 = []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
)

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix8AVX512ProbeSizes64 {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: forwardRadix8AVX512Complex64, Inverse: inverseRadix8AVX512Complex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX512,
			RankLevel:   fftypes.SIMDSSE2,
			Signature:   "dit" + itoa(size) + "_radix8ladder_avx512",
			Priority:    radix8AVX512ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex64,
		})
	}

	for _, size := range radix8AVX512ProbeSizes128 {
		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: forwardRadix8AVX512Complex128, Inverse: inverseRadix8AVX512Complex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDAVX512,
			RankLevel:   fftypes.SIMDSSE2,
			Signature:   "dit" + itoa(size) + "_radix8ladder_avx512",
			Priority:    radix8AVX512ProbePriority,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex128,
		})
	}
}
