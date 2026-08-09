//go:build amd64 && !purego

package algofft

import (
	"math/cmplx"
	"testing"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/planner"
)

func TestDirectionalWisdomPlanRoundTrip(t *testing.T) {
	const (
		n          = 512
		forwardSig = "dit512_radix4_avx2"
		inverseSig = "dit512_radix8ladder_avx2"
	)

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasSSE3:      true,
		HasAVX2:      true,
		HasFMA:       true,
	}
	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:          n,
			Precision:     uint8(PrecisionComplex128),
			CPUFeatures:   planner.CPUFeatureMask(true, true, true, false, false),
			CPUIdentifier: cpu.WisdomCPUIdentifier(features),
		},
		Algorithm: forwardSig + "/" + inverseSig,
		Timestamp: time.Now(),
	})

	plan, err := newPlanWithFeatures[complex128](n, features, PlanOptions{Wisdom: wisdom})
	if err != nil {
		t.Fatalf("newPlanWithFeatures: %v", err)
	}

	if plan.Algorithm() != forwardSig+"/"+inverseSig ||
		plan.ForwardAlgorithm() != forwardSig || plan.InverseAlgorithm() != inverseSig {
		t.Fatalf("algorithms = %q (%q, %q), want directional binding",
			plan.Algorithm(), plan.ForwardAlgorithm(), plan.InverseAlgorithm())
	}

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i%17)/17, float64(i%13)/13)
	}
	freq := make([]complex128, n)
	got := make([]complex128, n)

	if err := plan.Forward(freq, src); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if err := plan.Inverse(got, freq); err != nil {
		t.Fatalf("Inverse: %v", err)
	}

	for i := range got {
		if diff := cmplx.Abs(got[i] - src[i]); diff > 1e-11 {
			t.Fatalf("round trip[%d] diff = %g, want <= 1e-11", i, diff)
		}
	}

	if allocs := testing.AllocsPerRun(100, func() {
		_ = plan.Forward(freq, src)
		_ = plan.Inverse(got, freq)
	}); allocs != 0 {
		t.Fatalf("directional forward+inverse allocated %.2f times, want 0", allocs)
	}

	clone := plan.Clone()
	if clone.Algorithm() != plan.Algorithm() ||
		clone.ForwardAlgorithm() != plan.ForwardAlgorithm() ||
		clone.InverseAlgorithm() != plan.InverseAlgorithm() {
		t.Fatalf("clone algorithms = %q (%q, %q), want %q (%q, %q)",
			clone.Algorithm(), clone.ForwardAlgorithm(), clone.InverseAlgorithm(),
			plan.Algorithm(), plan.ForwardAlgorithm(), plan.InverseAlgorithm())
	}

	plan.Close()
	if clone.Algorithm() != forwardSig+"/"+inverseSig {
		t.Fatalf("clone Algorithm() after original Close = %q, want directional binding", clone.Algorithm())
	}
	if err := clone.Forward(freq, src); err != nil {
		t.Fatalf("clone Forward after original Close: %v", err)
	}
}
