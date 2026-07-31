package kernels

import (
	"math"
	"testing"
)

// A frozen synthetic workload for scripts/bench_gated.sh to use as its quiet-
// window canary.
//
// The canary's job is to measure the MACHINE -- thermal headroom and contention
// -- so that a candidate group is only ever timed inside a verified-quiet
// window. That job requires the canary's own runtime to be a constant. The
// previous canary was `BenchmarkDITComplex128/Size256/Radix16/Forward`, a real
// codelet in the very package the sweep tunes, so every successful
// optimisation moved the reference the gate is calibrated against. GOOD went
// stale twice that way, and it fails in the dangerous direction: a canary that
// has quietly got faster makes the gate too permissive, so contaminated
// windows are accepted as clean and the sweep reports numbers it should have
// rejected.
//
// So this workload is deliberately self-contained: a plain iterative radix-2
// DIT written out here in float32 pairs, calling nothing in the library and
// sharing no code with any kernel. It is representative enough to track the
// same throttling the real cells see (same broad mix of float multiply-add and
// L1-resident strided access) while being immune to anything the tuning work
// does.
//
// DO NOT OPTIMISE THIS. It is not a kernel and its speed is not a goal --
// changing it silently invalidates every GOOD value calibrated against it, and
// hence the accept/reject decision behind every measurement in
// docs/CODELET_BENCHMARKS.md. If it must change, treat that as a
// recalibration: re-derive GOOD on each machine and say so in
// scripts/bench_gated.sh.

const canaryN = 256

// canarySink defeats dead-code elimination without perturbing the timed loop.
var canarySink float32

// BenchmarkGateCanary is the quiet-window reference for the gated sweep.
func BenchmarkGateCanary(b *testing.B) {
	// Setup is untimed and deterministic: a fixed input and the twiddle
	// tables, built from math.Cos/Sin rather than internal/math so that this
	// file has no library dependency at all.
	srcRe := make([]float32, canaryN)
	srcIm := make([]float32, canaryN)

	for i := range canaryN {
		phase := 2 * math.Pi * float64(i) * 7 / canaryN
		srcRe[i] = float32(math.Cos(phase))
		srcIm[i] = float32(math.Sin(phase))
	}

	twRe := make([]float32, canaryN/2)
	twIm := make([]float32, canaryN/2)

	for i := range canaryN / 2 {
		phase := -2 * math.Pi * float64(i) / canaryN
		twRe[i] = float32(math.Cos(phase))
		twIm[i] = float32(math.Sin(phase))
	}

	// Bit-reversal permutation, computed here for the same no-dependency
	// reason. canaryN is a power of two, so this is the plain reversal.
	rev := make([]int, canaryN)

	bits := 0
	for 1<<bits < canaryN {
		bits++
	}

	for i := range canaryN {
		r := 0
		for bit := range bits {
			if i&(1<<bit) != 0 {
				r |= 1 << (bits - 1 - bit)
			}
		}

		rev[i] = r
	}

	workRe := make([]float32, canaryN)
	workIm := make([]float32, canaryN)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		// Reload from the fixed input every iteration. Without this the
		// transform would compound into overflow, and Inf/NaN timing is not
		// the machine state we are trying to observe.
		for i := range canaryN {
			workRe[i] = srcRe[rev[i]]
			workIm[i] = srcIm[rev[i]]
		}

		for m := 1; m < canaryN; m <<= 1 {
			step := canaryN / (2 * m)

			for start := 0; start < canaryN; start += 2 * m {
				for j := range m {
					wr := twRe[j*step]
					wi := twIm[j*step]

					a := start + j
					c := a + m

					tr := workRe[c]*wr - workIm[c]*wi
					ti := workRe[c]*wi + workIm[c]*wr

					workRe[c] = workRe[a] - tr
					workIm[c] = workIm[a] - ti
					workRe[a] += tr
					workIm[a] += ti
				}
			}
		}

		canarySink += workRe[1]
	}
}
