//go:build fftprobe

package kernels

// Measurement harness for the three six-step codelets, retired from the
// registry 2026-08-02. Built only under `-tags fftprobe`, so no ordinary
// build, test or benchmark sees any of it.
//
// Why these left the registry. Two canary-gated pure-Go sweeps put them
// behind the incumbent at every size, in both precisions and both
// directions, on two microarchitectures:
//
//	                                i7-1255U        Xeon Gold 5218
//	  4096   sixstep      c64    1.428 / 1.489    1.594 / 1.719
//	                      c128   1.488 / 1.896    1.736 / 1.833
//	  8192   sixstep64x128 c64   1.706 / 1.914    2.032 / 2.168
//	                      c128   2.058 / 2.091    2.191 / 2.348
//	  16384  sixstep      c64    1.892 / 2.111    2.097 / 1.818
//	                      c128   2.202 / 2.063    2.087 / 2.324
//
// (Ratios against the group incumbent, which is `dit<N>_radix8ladder_generic`
// at 4096/8192/16384. Twenty-four cells, no win anywhere; they lose even to
// the `radix4`/`radix4_then2` rows the ladder replaced.)
//
// Why that is a fair verdict on the codelets, and NOT on six-step. This is the
// distinction §2.2's "a poor implementation disqualifies the file, not the
// algorithm" exists to protect, and the two halves of six-step in this tree
// fall on opposite sides of it:
//
//   - These codelets call the tuned pure-Go radix-4 leaves
//     (forwardDIT64Radix4Complex64 and friends), so on a purego sweep both
//     arms are scalar and the comparison is like-for-like. That is a loss on
//     merit, and it is what retired these rows.
//   - The *strategy* kernel `ForwardSixStepComplex64` is a different
//     implementation, and its 17-35x plan-level loss is confounded: it
//     hardwires its row passes to the generic `stockhamForward` (87% of its
//     cost at n = 65536), so on a SIMD build it is a scalar kernel racing AVX2
//     codelets. That number must not be cited as a verdict on six-step, and
//     the family stays open under the Phase 3 item that owns the row binding.
//
// So a six-step decomposition may still be worth having. What is settled is
// that *these three codelets, as written*, are not the way to get it — which
// is exactly the state the fftprobe tag is for: out of every production build
// and every registry lookup, still compiled, still correctness-tested here,
// and still re-measurable on a host with a different cache geometry.
//
// Take the number with:
//
//	PATH=/usr/local/go/bin:$PATH GOFLAGS=-tags=fftprobe,purego GOOD=<canary floor> \
//	  taskset -c 0 ./scripts/bench_gated.sh 4096 8192 16384
//	scripts/bench_gated_analyze.sh benchmarks/gated

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// sixStepProbePriority keeps the arms last in the pure-Go tier, so the sweep
// reports them against the tuned incumbent rather than against each other.
const sixStepProbePriority = 1

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	type probe64 struct {
		size             int
		signature        string
		forward, inverse fftypes.CodeletFunc[complex64]
	}

	type probe128 struct {
		size             int
		signature        string
		forward, inverse fftypes.CodeletFunc[complex128]
	}

	for _, p := range []probe64{
		{4096, "dit4096_sixstep_generic", forwardDIT4096SixStepComplex64, inverseDIT4096SixStepComplex64},
		{8192, "dit8192_sixstep64x128_generic", forwardDIT8192SixStep64x128Complex64, inverseDIT8192SixStep64x128Complex64},
		{16384, "dit16384_sixstep_generic", forwardDIT16384SixStepComplex64, inverseDIT16384SixStepComplex64},
	} {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: p.size, Forward: p.forward, Inverse: p.inverse,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature:  p.signature,
			Priority:   sixStepProbePriority,
			KernelType: fftypes.KernelTypeDIT,
		})
	}

	for _, p := range []probe128{
		{4096, "dit4096_sixstep_generic", forwardDIT4096SixStepComplex128, inverseDIT4096SixStepComplex128},
		{8192, "dit8192_sixstep64x128_generic", forwardDIT8192SixStep64x128Complex128, inverseDIT8192SixStep64x128Complex128},
		{16384, "dit16384_sixstep_generic", forwardDIT16384SixStepComplex128, inverseDIT16384SixStepComplex128},
	} {
		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: p.size, Forward: p.forward, Inverse: p.inverse,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNone,
			Signature:  p.signature,
			Priority:   sixStepProbePriority,
			KernelType: fftypes.KernelTypeDIT,
		})
	}
}
