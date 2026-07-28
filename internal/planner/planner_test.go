package planner

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// TestEstimatePlanPowerOf2 tests EstimatePlan with power-of-2 sizes.
// Note: Avoiding hard expectations on specific strategies due to EstimatePlan's
// complex logic with codelet registry and wisdom cache. Instead, test observable behavior.
func TestEstimatePlanPowerOf2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
	}{
		{"Size 8", 8},
		{"Size 16", 16},
		{"Size 1024", 1024},
		{"Size 2048", 2048},
		{"Size 4096", 4096},
		{"Size 65536", 65536},
	}

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			estimate := EstimatePlan[complex64](tt.size, features, nil, KernelAuto)

			// Verify the estimate is non-zero and has valid strategy
			if estimate.Strategy == 0 {
				t.Errorf("EstimatePlan(%d) returned zero strategy", tt.size)
			}

			if estimate.Algorithm == "" {
				t.Errorf("EstimatePlan(%d) returned empty algorithm name", tt.size)
			}

			// For forced strategy, ensure it respects the force
			forcedEstimate := EstimatePlan[complex64](tt.size, features, nil, KernelDIT)
			if forcedEstimate.Strategy != KernelDIT {
				t.Errorf("EstimatePlan(%d, forced=DIT) strategy = %v, want KernelDIT", tt.size, forcedEstimate.Strategy)
			}
		})
	}
}

// TestEstimatePlanWithForcedStrategy tests EstimatePlan with forced kernel strategy.
func TestEstimatePlanWithForcedStrategy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		size          int
		forcedStrat   KernelStrategy
		expectedStrat KernelStrategy
	}{
		{"Force DIT on 2048", 2048, KernelDIT, KernelDIT},
		{"Force Stockham on 256", 256, KernelStockham, KernelStockham},
		{"Force Stockham on 2048", 2048, KernelStockham, KernelStockham},
	}

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			estimate := EstimatePlan[complex64](tt.size, features, nil, tt.forcedStrat)

			if estimate.Strategy != tt.expectedStrat {
				t.Errorf("EstimatePlan(%d, forced=%v) strategy = %v, want %v",
					tt.size, tt.forcedStrat, estimate.Strategy, tt.expectedStrat)
			}
		})
	}
}

// TestEstimatePlanNonPowerOf2 tests EstimatePlan with non-power-of-2 sizes.
func TestEstimatePlanNonPowerOf2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		size            int
		expectBluestein bool
	}{
		{"Size 1000 (highly composite)", 1000, false}, // 2³ × 5³ - not bluestein
		{"Size 1500 (highly composite)", 1500, false}, // 2² × 3 × 5³ - not bluestein
		{"Size 3072 (highly composite)", 3072, false}, // 2¹⁰ × 3 - not bluestein
		{"Size 1001 (not composite)", 1001, true},     // 7 × 11 × 13 - bluestein required
	}

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			estimate := EstimatePlan[complex64](tt.size, features, nil, KernelAuto)

			if tt.expectBluestein {
				if estimate.Algorithm != "bluestein" {
					t.Errorf("EstimatePlan(%d) algorithm = %q, want \"bluestein\" for non-5-smooth number",
						tt.size, estimate.Algorithm)
				}
			} else {
				// Highly composite numbers use fallback strategies, not bluestein
				if estimate.Algorithm == "bluestein" {
					t.Errorf("EstimatePlan(%d) algorithm = \"bluestein\", but %d is 5-smooth",
						tt.size, tt.size)
				}
			}
		})
	}
}

// TestEstimatePlanComplex128 tests EstimatePlan with complex128 precision.
func TestEstimatePlanComplex128(t *testing.T) {
	t.Parallel()

	features := cpu.Features{
		Architecture: "amd64",
		HasAVX2:      true,
	}

	estimate := EstimatePlan[complex128](1024, features, nil, KernelAuto)

	if estimate.Algorithm != "dit_fallback" {
		t.Errorf("EstimatePlan[complex128](1024) algorithm = %q, want \"dit_fallback\"", estimate.Algorithm)
	}
}

// TestEstimatePlanWithWisdom tests EstimatePlan with wisdom cache fallback.
func TestEstimatePlanWithWisdom(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        512,
			Precision:   0,
			CPUFeatures: CPUFeatureMask(true, false, true, false, false),
		},
		Algorithm: "stockham",
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasAVX2:      true,
	}

	estimate := EstimatePlan[complex64](512, features, wisdom, KernelAuto)

	// Wisdom recommends stockham for size 512
	if estimate.Strategy != KernelStockham {
		t.Errorf("EstimatePlan with wisdom: strategy = %v, want KernelStockham", estimate.Strategy)
	}
}

// TestEstimatePlanWisdomOverriddenByForce tests that forced strategy overrides wisdom.
func TestEstimatePlanWisdomOverriddenByForce(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        512,
			Precision:   0,
			CPUFeatures: CPUFeatureMask(true, false, true, false, false),
		},
		Algorithm: "stockham",
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasAVX2:      true,
	}

	estimate := EstimatePlan[complex64](512, features, wisdom, KernelDIT)

	// Forced strategy should override wisdom
	if estimate.Strategy != KernelDIT {
		t.Errorf("EstimatePlan forced override: strategy = %v, want KernelDIT", estimate.Strategy)
	}
}

// TestResolveWisdomRejectsUnsupportedCodelet verifies that wisdom cannot bind
// a codelet whose SIMD level the CPU does not support (e.g. an AVX2 codelet on
// a CPU with FMA masked off — the wisdom feature mask alone does not
// distinguish this case).
// dummyCodelet is a stand-in codelet body for registry entries in tests.
func dummyCodelet[T Complex](dst, src, twiddle, scratch []T) bool { return true }

func TestResolveWisdomRejectsUnsupportedCodelet(t *testing.T) {
	t.Parallel()

	// Unique size and signature so the global registry entry cannot interfere
	// with other tests.
	const (
		size = 1 << 19
		sig  = "wisdomtest_avx2"
	)

	registry.GetRegistry[complex64]().Register(registry.CodeletEntry[complex64]{
		Size:      size,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: fftypes.SIMDAVX2,
		Signature: sig,
		Priority:  1,
	})

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        size,
			Precision:   0,
			CPUFeatures: CPUFeatureMask(true, false, true, false, false),
		},
		Algorithm: sig,
	})

	// AVX2 present but FMA masked off: the AVX2 codelet must not be bound.
	noFMA := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasAVX2:      true,
	}

	algorithm, ok := wisdomAlgorithm[complex64](size, noFMA, wisdom)
	if !ok {
		t.Fatalf("wisdomAlgorithm did not find the stored entry")
	}

	est := bindWisdomCodelet[complex64](size, noFMA, algorithm, KernelAuto)
	if est != nil {
		t.Errorf("bindWisdomCodelet bound CPU-incompatible codelet: est=%v", est)
	}

	// Sanity check: with FMA available the same wisdom entry binds the codelet.
	withFMA := noFMA
	withFMA.HasFMA = true

	est = bindWisdomCodelet[complex64](size, withFMA, algorithm, KernelAuto)
	if est == nil {
		t.Fatalf("bindWisdomCodelet did not bind supported codelet")
	}

	if est.Algorithm != sig {
		t.Errorf("bindWisdomCodelet algorithm = %q, want %q", est.Algorithm, sig)
	}
}

// TestEstimatePlanWisdomOutranksRegistry pins the ordering that makes wisdom
// usable at all: a wisdom entry naming a codelet signature must win over the
// registry's static priority order for the same size. Before this, the registry
// was consulted first and hit for every size that has a codelet — i.e. every
// size for which a signature can exist — so signature-level wisdom was
// unreachable in practice.
func TestEstimatePlanWisdomOutranksRegistry(t *testing.T) {
	t.Parallel()

	const (
		size    = 1 << 20
		bestSig = "wisdomrank_best"
		pinned  = "wisdomrank_pinned"
	)

	reg := registry.GetRegistry[complex64]()
	for _, e := range []struct {
		sig      string
		priority int
	}{{bestSig, 90}, {pinned, 10}} {
		reg.Register(registry.CodeletEntry[complex64]{
			Size:      size,
			Forward:   dummyCodelet[complex64],
			Inverse:   dummyCodelet[complex64],
			Algorithm: KernelDIT,
			SIMDLevel: fftypes.SIMDNone,
			Signature: e.sig,
			Priority:  e.priority,
		})
	}

	features := cpu.Features{Architecture: "amd64", HasSSE2: true}

	// No wisdom: the registry's own order decides.
	if got := EstimatePlan[complex64](size, features, nil, KernelAuto); got.Algorithm != bestSig {
		t.Errorf("without wisdom: algorithm = %q, want %q", got.Algorithm, bestSig)
	}

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        size,
			Precision:   PrecisionComplex64,
			CPUFeatures: CPUFeatureMask(true, false, false, false, false),
		},
		Algorithm: pinned,
	})

	got := EstimatePlan[complex64](size, features, wisdom, KernelAuto)
	if got.Algorithm != pinned {
		t.Errorf("with wisdom: algorithm = %q, want %q", got.Algorithm, pinned)
	}

	if got.ForwardCodelet == nil || got.InverseCodelet == nil {
		t.Error("wisdom-bound estimate has no codelet bindings")
	}
}

// TestEstimatePlanStrategyWisdomYieldsToCodelet is the other half of the
// ordering: a wisdom entry naming a *strategy* must not displace a codelet. The
// measurement behind such an entry (internal/fft.benchmarkStrategy) only ever
// times the kernel path, so it carries no evidence about the codelet.
func TestEstimatePlanStrategyWisdomYieldsToCodelet(t *testing.T) {
	t.Parallel()

	const (
		size = 1 << 21
		sig  = "wisdomstrat_codelet"
	)

	registry.GetRegistry[complex64]().Register(registry.CodeletEntry[complex64]{
		Size:      size,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: sig,
		Priority:  1,
	})

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        size,
			Precision:   PrecisionComplex64,
			CPUFeatures: CPUFeatureMask(true, false, false, false, false),
		},
		Algorithm: algoStockham,
	})

	features := cpu.Features{Architecture: "amd64", HasSSE2: true}

	got := EstimatePlan[complex64](size, features, wisdom, KernelAuto)
	if got.Algorithm != sig {
		t.Errorf("algorithm = %q, want the codelet %q", got.Algorithm, sig)
	}
}

// TestEstimatePlanWisdomSkipsDisabledCodelet verifies that the stale-entry
// guard in registry.LookupBySignature is actually reachable: a wisdom entry
// naming a codelet that has since been disabled (negative priority) must not
// resurrect it.
func TestEstimatePlanWisdomSkipsDisabledCodelet(t *testing.T) {
	t.Parallel()

	const (
		size       = 1 << 22
		enabledSig = "wisdomdisabled_enabled"
		staleSig   = "wisdomdisabled_stale"
	)

	reg := registry.GetRegistry[complex64]()
	for _, e := range []struct {
		sig      string
		priority int
	}{{enabledSig, 5}, {staleSig, -1}} {
		reg.Register(registry.CodeletEntry[complex64]{
			Size:      size,
			Forward:   dummyCodelet[complex64],
			Inverse:   dummyCodelet[complex64],
			Algorithm: KernelDIT,
			SIMDLevel: fftypes.SIMDNone,
			Signature: e.sig,
			Priority:  e.priority,
		})
	}

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        size,
			Precision:   PrecisionComplex64,
			CPUFeatures: CPUFeatureMask(true, false, false, false, false),
		},
		Algorithm: staleSig,
	})

	features := cpu.Features{Architecture: "amd64", HasSSE2: true}

	got := EstimatePlan[complex64](size, features, wisdom, KernelAuto)
	if got.Algorithm != enabledSig {
		t.Errorf("algorithm = %q, want the enabled codelet %q", got.Algorithm, enabledSig)
	}
}

// TestHasCodelet tests the HasCodelet function.
func TestHasCodelet(t *testing.T) {
	t.Parallel()

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	// Initially no codelets registered
	has := HasCodelet[complex64](256, features)
	if has {
		t.Error("HasCodelet should return false when no codelets registered")
	}
}

// TestMixedRadixEligible locks in the measured win gate for lengths with
// factors 7/11 (see mixedRadix7And11Wins for the benchmark rationale).
func TestMixedRadixEligible(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want bool
	}{
		// Non-smooth lengths are never eligible.
		{n: 13, want: false},
		{n: 26, want: false},
		{n: 1001, want: false},
		// 5-smooth lengths are always eligible (incumbent behavior).
		{n: 12, want: true},
		{n: 15, want: true},
		{n: 480, want: true},
		// Factors 7/11 with power-of-two part >= 8: measured wins.
		{n: 56, want: true},
		{n: 448, want: true},
		{n: 616, want: true},
		{n: 704, want: true},
		{n: 1344, want: true},
		// Power-of-two part 2 or 4: the pad-ratio rule decides, same as for
		// odd lengths. 44 pads to 2.9n and 308 to 3.3n (wins); 14, 28, 462
		// and 924 pad to ~2.3n and stay on Bluestein.
		{n: 14, want: false},
		{n: 28, want: false},
		{n: 44, want: true},
		{n: 308, want: true},
		{n: 462, want: false},
		{n: 924, want: false},
		// The large audio-rate shapes the gate previously sent to Bluestein.
		{n: 1100, want: true},
		{n: 2156, want: true},
		{n: 4900, want: true},
		{n: 6300, want: true},
		{n: 8820, want: true},
		{n: 22050, want: true},
		{n: 44100, want: true},
		// Odd: eligible when the Bluestein pad is >= ~2.5n.
		{n: 7, want: false},
		{n: 11, want: true},
		{n: 35, want: true},
		{n: 63, want: false},
		{n: 77, want: true},
		{n: 121, want: false},
		{n: 231, want: false},
		{n: 385, want: true},
		{n: 847, want: false},
		{n: 2401, want: true},
	}

	for _, tt := range tests {
		got := MixedRadixEligible(tt.n)
		if got != tt.want {
			t.Errorf("MixedRadixEligible(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}
