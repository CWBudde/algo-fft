package transform

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// The policy knob is process-global, so these tests must not run in parallel
// with anything that builds plans. They restore the default before returning.

func withOverride(t *testing.T, mode PackedOverride) {
	t.Helper()
	SetPackedStockhamOverride(mode)
	t.Cleanup(func() { SetPackedStockhamOverride(PackedOverrideDefault) })
}

func TestPackedStockhamOverride(t *testing.T) {
	features := cpu.DetectFeatures()

	t.Run("ForceOn", func(t *testing.T) {
		withOverride(t, PackedOverrideForceOn)

		// n = 4 is below every threshold in the table, so only the override
		// can be answering here.
		for _, wide := range []bool{false, true} {
			if !PackedStockhamEnabled(4, wide, features) {
				t.Errorf("wide=%v: ForceOn did not enable the packed route", wide)
			}
		}
	})

	t.Run("ForceOff", func(t *testing.T) {
		withOverride(t, PackedOverrideForceOff)

		for _, wide := range []bool{false, true} {
			if PackedStockhamEnabled(1<<22, wide, features) {
				t.Errorf("wide=%v: ForceOff did not disable the packed route", wide)
			}
		}
	})

	t.Run("DefaultRestored", func(t *testing.T) {
		if got := PackedStockhamOverride(); got != PackedOverrideDefault {
			t.Fatalf("override leaked between tests: %v", got)
		}
	})
}

// TestPackedStockhamTierTable checks PackedStockhamEnabled against the table
// for the tier this build resolves to, on both sides of each threshold.
func TestPackedStockhamTierTable(t *testing.T) {
	features := cpu.DetectFeatures()
	tier := packedTierFor(features)

	if !packedBuildHasSIMDKernels && tier != tierGeneric {
		t.Fatalf("pure-Go build resolved to tier %d, want tierGeneric", tier)
	}

	sizes := []int{4, 1024, 1 << 16, 1 << 17, 1 << 19, 1 << 20, 1 << 22}

	for _, wide := range []bool{false, true} {
		column := 0
		if wide {
			column = 1
		}

		threshold := packedStockhamMinSize[tier][column]

		for _, n := range sizes {
			got := PackedStockhamEnabled(n, wide, features)
			want := n >= threshold

			if got != want {
				t.Errorf("tier=%d n=%d wide=%v: enabled=%v, want %v (threshold %d)",
					tier, n, wide, got, want, threshold)
			}
		}
	}
}

// TestPackedStockhamThresholdsAreOrdered pins the shape of the measured AVX2
// row: complex128 crosses over strictly earlier than complex64, which is the
// whole reason the table has a precision axis. A future retune that inverts
// this ordering is far more likely to be a mistake than a finding.
func TestPackedStockhamThresholdsAreOrdered(t *testing.T) {
	t.Parallel()

	avx2 := packedStockhamMinSize[tierAVX2]

	if avx2[1] >= avx2[0] {
		t.Errorf("AVX2 thresholds: complex128 %d should be below complex64 %d", avx2[1], avx2[0])
	}

	if got := packedStockhamMinSize[tierGeneric]; got != [2]int{4, 4} {
		t.Errorf("generic tier = %v, want {4, 4}: pure-Go builds must keep taking the packed route", got)
	}
}

// TestPackedTierForIgnoresFeaturesOnPureGo is the regression guard for the trap
// that makes this a tier selector rather than a feature check:
// internal/cpu/detect_amd64.go carries no `!purego` build tag, so a purego
// amd64 build still reports HasAVX2. Keying on features alone would silently
// change purego behaviour (PLAN.md 2.1 rule 4).
func TestPackedTierForIgnoresFeaturesOnPureGo(t *testing.T) {
	t.Parallel()

	loud := cpu.Features{HasSSE2: true, HasAVX2: true, HasAVX512: true, HasNEON: true}

	got := packedTierFor(loud)

	if packedBuildHasSIMDKernels {
		if got == tierGeneric {
			t.Error("SIMD build resolved AVX-512 features to tierGeneric")
		}
	} else if got != tierGeneric {
		t.Errorf("pure-Go build resolved to tier %d despite no SIMD kernels, want tierGeneric", got)
	}
}

// TestPackedTierForForceGeneric pins that ForceGeneric reaches this decision;
// it is the documented way to disable SIMD and must not leave a SIMD-tier
// threshold in force.
func TestPackedTierForForceGeneric(t *testing.T) {
	t.Parallel()

	forced := cpu.Features{HasSSE2: true, HasAVX2: true, ForceGeneric: true}

	if got := packedTierFor(forced); got != tierGeneric {
		t.Errorf("ForceGeneric resolved to tier %d, want tierGeneric", got)
	}
}
