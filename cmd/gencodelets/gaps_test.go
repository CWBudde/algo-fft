package main

import (
	"runtime"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	_ "github.com/cwbudde/algo-fft/internal/kernels" // registers every codelet
	"github.com/cwbudde/algo-fft/internal/registry"
)

// simdLevels maps the spec table's level identifier to the enum the registry
// gates on.
//
//nolint:gochecknoglobals // test fixture
var simdLevels = map[string]fftypes.SIMDLevel{
	"SIMDNone":   fftypes.SIMDNone,
	"SIMDSSE2":   fftypes.SIMDSSE2,
	"SIMDSSE3":   fftypes.SIMDSSE3,
	"SIMDAVX2":   fftypes.SIMDAVX2,
	"SIMDAVX512": fftypes.SIMDAVX512,
	"SIMDNEON":   fftypes.SIMDNEON,
}

// features builds the CPU a host profile describes.
func (h hostProfile) features() cpu.Features {
	f := cpu.Features{Architecture: h.Arch}

	for _, l := range h.Levels {
		switch l {
		case "SIMDSSE2":
			f.HasSSE2 = true
		case "SIMDSSE3":
			f.HasSSE2, f.HasSSE3 = true, true
		case "SIMDAVX2":
			f.HasAVX2, f.HasFMA = true, true
		case "SIMDAVX512":
			f.HasAVX512 = true
		case "SIMDNEON":
			f.HasNEON = true
		}
	}

	return f
}

// TestResolveMatchesLiveRegistry is the gate that makes the coverage table a
// fact rather than a model: the generator replays registry.Lookup from the
// spec table, so a change to Register's ordering, to the SIMDLevel gate, or to
// the disabled-row rule must show up here rather than silently making the
// generated numbers describe a registry that no longer exists.
//
// The comparison is restricted to the tiers this build actually registers
// (arm64 has no AVX2 rows, -tags purego has none at all), taken from the
// registry itself rather than assumed from GOARCH.
func TestResolveMatchesLiveRegistry(t *testing.T) {
	registered := registeredLevels(t)

	for _, h := range hostProfiles {
		if h.Arch != "any" && h.Arch != runtime.GOARCH {
			continue
		}

		got := lookupAll(registry.Registry64, h.features())

		for _, size := range registry.Registry64.Sizes() {
			want, ok := h.resolveIn(64, size, registered)
			if !ok {
				if got[size] != "" {
					t.Errorf("%s n=%d: registry serves %q, the generator expects nothing",
						h.Name, size, got[size])
				}

				continue
			}

			if got[size] != want.Signature {
				t.Errorf("%s n=%d: registry serves %q, the generator expects %q",
					h.Name, size, got[size], want.Signature)
			}
		}
	}
}

// TestEveryTierHasAHostProfile fails when a new SIMD column appears with no
// host that tops out there — the coverage table would then silently omit the
// machine the new tier was written for.
func TestEveryTierHasAHostProfile(t *testing.T) {
	tops := map[string]int{}
	for _, h := range hostProfiles {
		tops[h.Top]++
	}

	for _, level := range simdColumns {
		if tops[level] != 1 {
			t.Errorf("%s is the top ISA of %d host profiles, want exactly 1", level, tops[level])
		}
	}

	// A host must be able to execute its own top level, and every level it
	// claims must be a real column.
	for _, h := range hostProfiles {
		found := false

		for _, l := range h.Levels {
			if levelOrder(l) < 0 {
				t.Errorf("%s claims unknown level %q", h.Name, l)
			}

			if l == h.Top {
				found = true
			}
		}

		if !found {
			t.Errorf("%s tops out at %s but does not list it", h.Name, h.Top)
		}
	}
}

// TestGapsAreRelativeToRealCoverage checks the claim the gap table makes: a
// listed gap is a size some other tier covers at the same precision. A gap
// nothing covers is not a gap, it is the size-generic tier doing its job, and
// listing it would turn the section into noise.
func TestGapsAreRelativeToRealCoverage(t *testing.T) {
	for _, prec := range []int{64, 128} {
		covered := map[int]bool{}
		for _, size := range registeredSizes(prec) {
			covered[size] = true
		}

		for _, level := range simdColumns {
			for _, size := range tierSizes(prec, level) {
				if !covered[size] {
					t.Errorf("%s complex%d covers n=%d, which registeredSizes omits", level, prec, size)
				}
			}

			for _, size := range precOnly(level, prec, otherPrec(prec)) {
				if !covered[size] {
					t.Errorf("%s complex%d-only n=%d is not a registered size", level, prec, size)
				}
			}
		}
	}
}

// TestRankedSpecsSkipsDisabledRows guards the one place the replay could
// silently over-report coverage: a negative-priority row must not answer a
// lookup, because the registry skips it.
func TestRankedSpecsSkipsDisabledRows(t *testing.T) {
	for _, s := range codeletSpecs {
		if s.Priority >= 0 {
			continue
		}

		for _, got := range rankedSpecs(s.Prec, s.Size) {
			if got.Signature == s.Signature {
				t.Errorf("disabled %s (complex%d n=%d) is ranked as selectable",
					s.Signature, s.Prec, s.Size)
			}
		}
	}
}

func otherPrec(prec int) int {
	if prec == 64 {
		return 128
	}

	return 64
}

// resolveIn is resolve restricted to the tiers a build registers.
func (h hostProfile) resolveIn(prec, size int, registered map[string]bool) (codeletSpec, bool) {
	supported := map[string]bool{}
	for _, l := range h.Levels {
		supported[l] = true
	}

	for _, s := range rankedSpecs(prec, size) {
		if supported[s.SIMDLevel] && registered[s.SIMDLevel] {
			return s, true
		}
	}

	return codeletSpec{}, false
}

// registeredLevels reports which SIMD levels this build actually put in the
// registry.
func registeredLevels(t *testing.T) map[string]bool {
	t.Helper()

	out := map[string]bool{}

	for _, size := range registry.Registry64.Sizes() {
		for _, e := range registry.Registry64.GetAllForSize(size) {
			for name, level := range simdLevels {
				if e.SIMDLevel == level {
					out[name] = true
				}
			}
		}
	}

	if len(out) == 0 {
		t.Fatal("no codelets registered at all — the comparison would be vacuous")
	}

	return out
}

// lookupAll resolves every registered size against one CPU.
func lookupAll(r *registry.CodeletRegistry[complex64], f cpu.Features) map[int]string {
	out := map[int]string{}

	for _, size := range r.Sizes() {
		if e := r.Lookup(size, f); e != nil {
			out[size] = e.Signature
		}
	}

	return out
}
