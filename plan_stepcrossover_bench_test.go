package algofft

import "testing"

// stepCrossoverSizes is the size band PLAN.md §1.2's crossover item leaves
// unmeasured. Below it sit the three sizes where six-step has registered
// codelet rows (4096, 8192, 16384), which the canary-gated registry sweep
// ranks directly; above it sits BenchmarkSquareAutoRule, whose floor is 2^18.
// The item's own estimate — "it is above 16384 on this host" — points exactly
// here, and nothing had measured it.
//
// The arms are the decompositions against the flat ladder they have to
// overtake. KernelDIT is that ladder: at a size with a registered codelet the
// forced-DIT plan binds it (every codelet carries Algorithm: KernelDIT, and
// tryRegistry declines any other forced strategy), and where the registry has
// nothing it takes internal/kernels/dit.go's hand-tuned size switch. So the
// DIT arm is the incumbent at every size here, not a generic baseline.
//
// KernelSixStep runs only at the perfect squares: resolveKernelStrategy falls
// a forced six-step back to the size heuristic at every other length, and
// squareArm's assertion that the force took would fire.
//
// There is deliberately no KernelEightStep arm. internal/kernels/eightstep.go
// is internal/kernels/sixstep.go with the names changed — same perfect-square
// rejection, same two TransposeSquare-bracketed Stockham row passes, no eighth
// step anywhere. An arm that is a renamed copy of another arm would make this
// table look like it covers a family it does not; the family's verdict rests
// on that diff rather than on a measurement.
//
// The KernelDIT arm is measured for information only. Retuning
// ditAutoThreshold against it is PLAN.md §1.4's item, not this one.
//
//nolint:gochecknoglobals // benchmark input table
var stepCrossoverSizes = []struct {
	n      int
	label  string
	square bool
}{
	{n: 1 << 14, label: "16384", square: true},   // 128x128
	{n: 1 << 15, label: "32768", square: false},  //
	{n: 1 << 16, label: "65536", square: true},   // 256x256
	{n: 1 << 17, label: "131072", square: false}, //
}

// BenchmarkStepCrossover measures where six-step and four-step overtake the
// flat radix ladder, in the band between the six-step codelet sizes and
// BenchmarkSquareAutoRule's 2^18 floor.
//
// It reuses squareArm/benchSquareArm deliberately: those already run the arms
// for one size adjacent in a single process, cache the plan per (precision,
// size, strategy) because plan construction dominates the transform at these
// lengths, seed random input so no arm times a denormal-heavy zero spectrum,
// and fail the benchmark if a forced strategy did not take.
//
// Run per PLAN.md §2.3: /usr/local/go/bin/go under `taskset -c 0`, -count=5,
// medians, and again with -tags purego.
func BenchmarkStepCrossover(b *testing.B) {
	for _, tc := range stepCrossoverSizes {
		arms := []KernelStrategy{KernelDIT, KernelStockham, KernelFourStep}
		if tc.square {
			arms = append(arms, KernelSixStep)
		}

		for _, dir := range []string{"fwd", "inv"} {
			for _, arm := range arms {
				name := tc.label + "/" + dir + "/" + arm.String()

				b.Run("complex64/"+name, func(b *testing.B) {
					benchSquareArm[complex64](b, tc.n, arm, dir == "fwd")
				})

				b.Run("complex128/"+name, func(b *testing.B) {
					benchSquareArm[complex128](b, tc.n, arm, dir == "fwd")
				})
			}
		}
	}
}
