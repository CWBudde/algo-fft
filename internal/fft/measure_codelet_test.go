package fft

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// measureTestCodelet is a stand-in codelet body: it does no transform, it only
// has to report success so the candidate can be timed.
func measureTestCodelet(dst, src, _, _ []complex64) bool {
	copy(dst, src)

	return true
}

// registerMeasureCodelets registers entries at a size no real codelet uses, so
// the candidate list for that size is exactly what these tests put there.
//
// WARNING: the codelet registry is process-global and append-only — there is no
// way to unregister. Every size passed here (1155 and 1331 below) is therefore
// identity-mapped for the whole test binary: any later transform at that length
// dispatches to measureTestCodelet, copies its input to its output and reports
// success. A correctness test that happens to pick one of those lengths fails in
// a full package run and passes under -run, which is a slow thing to diagnose.
// Keep the sizes here odd-looking, keep this list short, and pick new ones that
// no transform test would reach for.
func registerMeasureCodelets(t *testing.T, size int, entries ...registry.CodeletEntry[complex64]) {
	t.Helper()

	for _, entry := range entries {
		entry.Size = size
		entry.Forward = measureTestCodelet
		entry.Inverse = measureTestCodelet
		entry.Algorithm = fftypes.KernelDIT

		registry.GetRegistry[complex64]().Register(entry)
	}
}

func candidateNames(cands []measureCandidate[complex64]) []string {
	names := make([]string, 0, len(cands))
	for i := range cands {
		if cands[i].entry != nil {
			names = append(names, cands[i].algorithm)
		}
	}

	return names
}

// TestMeasureCandidatesIncludeCodelets is the regression behind
// "PlannerMeasure can pick a worse plan than PlannerEstimate": measurement used
// to time kernel strategies only, so a codelet could be discarded without ever
// having been compared with what replaced it. The quick mode times the
// registry's own winner; the deeper modes time every enabled codelet the CPU
// can run.
func TestMeasureCandidatesIncludeCodelets(t *testing.T) {
	t.Parallel()

	const (
		size    = 1155 // 3*5*7*11: mixed-radix eligible, no real codelet
		topSig  = "measurecand_top"
		altSig  = "measurecand_alt"
		offSig  = "measurecand_disabled"
		avx2Sig = "measurecand_avx2"
	)

	registerMeasureCodelets(
		t, size,
		registry.CodeletEntry[complex64]{Signature: topSig, Priority: 50, SIMDLevel: fftypes.SIMDNone},
		registry.CodeletEntry[complex64]{Signature: altSig, Priority: 10, SIMDLevel: fftypes.SIMDNone},
		registry.CodeletEntry[complex64]{Signature: offSig, Priority: -1, SIMDLevel: fftypes.SIMDNone},
		registry.CodeletEntry[complex64]{Signature: avx2Sig, Priority: 99, SIMDLevel: fftypes.SIMDAVX2},
	)

	// An SSE2-only CPU: the AVX2 codelet is registered but must never be timed.
	features := cpu.Features{Architecture: "amd64", HasSSE2: true}

	tests := []struct {
		name string
		mode PlannerMode
		want []string
	}{
		{"Measure times the registry winner", PlannerMeasure, []string{topSig}},
		{"Patient times every enabled codelet", PlannerPatient, []string{topSig, altSig}},
		{"Exhaustive times every enabled codelet", PlannerExhaustive, []string{topSig, altSig}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cands := measureCandidates[complex64](
				tt.mode, size, features, selectStrategiesToTest(tt.mode, size),
			)

			got := candidateNames(cands)
			if len(got) != len(tt.want) {
				t.Fatalf("codelet candidates = %v, want %v", got, tt.want)
			}

			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("codelet candidate %d = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestMeasureAndSelectBindsCodeletTwiddles verifies that a codelet winning the
// measurement is returned with its twiddle-preparation callbacks. Without them
// a codelet wanting a packed layout is handed the plain table, fails its own
// length check and silently runs the fallback kernel while the plan still
// reports the codelet's signature.
func TestMeasureAndSelectBindsCodeletTwiddles(t *testing.T) {
	t.Parallel()

	const (
		size = 1331 // 11^3: mixed-radix eligible, no real codelet
		sig  = "measuretwiddle_packed"
	)

	twiddleSize := func(int) int { return 4 }
	prepare := func(_ int, _ bool, _ []complex64) {}

	registerMeasureCodelets(t, size, registry.CodeletEntry[complex64]{
		Signature:      sig,
		Priority:       50,
		SIMDLevel:      fftypes.SIMDNone,
		TwiddleSize:    twiddleSize,
		PrepareTwiddle: prepare,
	})

	features := cpu.Features{Architecture: "amd64", HasSSE2: true}

	// The stand-in codelet returns instantly, so it wins over any real kernel.
	estimate := MeasureAndSelect[complex64](size, features, PlannerMeasure, nil, fftypes.KernelAuto)

	if estimate.Algorithm != sig {
		t.Fatalf("algorithm = %q, want the codelet %q", estimate.Algorithm, sig)
	}

	if estimate.ForwardTwiddleSize == nil || estimate.ForwardPrepareTwiddle == nil ||
		estimate.InverseTwiddleSize == nil || estimate.InversePrepareTwiddle == nil {
		t.Error("winning codelet was bound without its twiddle callbacks")
	}
}

func TestDirectionalEstimateBindsSeparateCodelets(t *testing.T) {
	t.Parallel()

	forward := registry.CodeletEntry[complex64]{
		Forward:        measureTestCodelet,
		Inverse:        measureTestCodelet,
		Algorithm:      fftypes.KernelDIT,
		Signature:      "direction_forward",
		TwiddleSize:    func(int) int { return 4 },
		PrepareTwiddle: func(int, bool, []complex64) {},
	}
	inverse := forward
	inverse.Signature = "direction_inverse"

	estimate := directionalEstimate(
		measureCandidate[complex64]{entry: &forward, algorithm: forward.Signature, strategy: fftypes.KernelDIT},
		measureCandidate[complex64]{entry: &inverse, algorithm: inverse.Signature, strategy: fftypes.KernelDIT},
	)

	if estimate.Algorithm != "direction_forward/direction_inverse" {
		t.Fatalf("algorithm = %q, want directional pair", estimate.Algorithm)
	}

	if estimate.ForwardCodelet == nil || estimate.InverseCodelet == nil ||
		estimate.ForwardPrepareTwiddle == nil || estimate.InversePrepareTwiddle == nil {
		t.Error("directional estimate did not bind both codelets and twiddle callbacks")
	}
}
