package fft

import (
	"testing"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// mockWisdomRecorder records wisdom entries for testing.
type mockWisdomRecorder struct {
	entries []planner.WisdomEntry
}

func (m *mockWisdomRecorder) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (string, bool) {
	for _, e := range m.entries {
		if e.Key.Size == size && e.Key.Precision == precision && e.Key.CPUFeatures == cpuFeatures {
			return e.Algorithm, true
		}
	}

	return "", false
}

func (m *mockWisdomRecorder) LookupWisdomForCPU(
	size int, precision uint8, cpuFeatures uint64, cpuIdentifier string,
) (string, bool) {
	for _, entry := range m.entries {
		if entry.Key.Size == size && entry.Key.Precision == precision &&
			entry.Key.CPUFeatures == cpuFeatures && entry.Key.CPUIdentifier == cpuIdentifier {
			return entry.Algorithm, true
		}
	}

	return "", false
}

func (m *mockWisdomRecorder) Store(entry planner.WisdomEntry) {
	m.entries = append(m.entries, entry)
}

func TestSelectStrategiesToTest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		mode     PlannerMode
		n        int
		expected []fftypes.KernelStrategy
	}{
		{
			name:     "Measure mode power-of-two",
			mode:     PlannerMeasure,
			n:        1024,
			expected: []fftypes.KernelStrategy{fftypes.KernelDIT, fftypes.KernelStockham},
		},
		{
			name:     "Patient mode power-of-two",
			mode:     PlannerPatient,
			n:        1024,
			expected: []fftypes.KernelStrategy{fftypes.KernelDIT, fftypes.KernelStockham, fftypes.KernelSixStep, fftypes.KernelSplitRadix, fftypes.KernelFourStep},
		},
		{
			name:     "Exhaustive mode power-of-two",
			mode:     PlannerExhaustive,
			n:        1024,
			expected: []fftypes.KernelStrategy{fftypes.KernelDIT, fftypes.KernelStockham, fftypes.KernelSixStep, fftypes.KernelSplitRadix, fftypes.KernelFourStep},
		},
		{
			name:     "Prime size uses Bluestein only",
			mode:     PlannerExhaustive,
			n:        17,
			expected: []fftypes.KernelStrategy{fftypes.KernelBluestein},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := selectStrategiesToTest(tt.mode, tt.n)
			if len(got) != len(tt.expected) {
				t.Errorf("selectStrategiesToTest(%v, %d) = %v, want %v", tt.mode, tt.n, got, tt.expected)
				return
			}

			for i := range got {
				if got[i] != tt.expected[i] {
					t.Errorf("selectStrategiesToTest(%v, %d)[%d] = %v, want %v", tt.mode, tt.n, i, got[i], tt.expected[i])
				}
			}
		})
	}
}

func TestGetMeasureConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		mode           PlannerMode
		expectedWarmup int
		expectedIters  int
		expectedTrials int
	}{
		{PlannerMeasure, 5, 30, 5},
		{PlannerPatient, 5, 50, 7},
		{PlannerExhaustive, 10, 100, 9},
		{PlannerEstimate, 3, 30, 5}, // fallback
	}

	for _, tt := range tests {
		config := getMeasureConfig(tt.mode)
		if config.warmup != tt.expectedWarmup {
			t.Errorf("getMeasureConfig(%v).warmup = %d, want %d", tt.mode, config.warmup, tt.expectedWarmup)
		}

		if config.iters != tt.expectedIters {
			t.Errorf("getMeasureConfig(%v).iters = %d, want %d", tt.mode, config.iters, tt.expectedIters)
		}

		if config.trials != tt.expectedTrials {
			t.Errorf("getMeasureConfig(%v).trials = %d, want %d", tt.mode, config.trials, tt.expectedTrials)
		}
	}
}

// TestBenchmarkStrategy tests the benchmarkCandidate function on a strategy.
func TestBenchmarkStrategy(t *testing.T) {
	t.Parallel()

	// Use explicit features with ForceGeneric to ensure pure-Go fallbacks are used.
	// This makes the test immune to race conditions from other tests modifying
	// global CPU detection state via SetForcedFeatures().
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}

	tests := []struct {
		name     string
		n        int
		strategy fftypes.KernelStrategy
	}{
		{"DIT 64", 64, fftypes.KernelDIT},
		{"Stockham 64", 64, fftypes.KernelStockham},
		{"DIT 256", 256, fftypes.KernelDIT},
		{"Stockham 256", 256, fftypes.KernelStockham},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			config := measureConfig{warmup: 1, iters: 3}
			elapsed := benchmarkCandidate(tt.n, features, measureCandidate[complex64]{strategy: tt.strategy}, config)

			// Should complete without panicking and return positive duration
			if elapsed <= 0 {
				t.Errorf("benchmarkCandidate returned %v, expected positive duration", elapsed)
			}
		})
	}
}

// TestMeasureAndSelect_RecordsToWisdom tests that MeasureAndSelect records wisdom entries.
func TestMeasureAndSelect_RecordsToWisdom(t *testing.T) {
	t.Parallel()

	// Use explicit features to avoid race conditions with other tests.
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}
	recorder := &mockWisdomRecorder{}

	// Run with PlannerMeasure mode
	estimate := MeasureAndSelect[complex64](
		256,
		features,
		PlannerMeasure,
		recorder,
		fftypes.KernelAuto,
	)

	// Should have recorded an entry
	if len(recorder.entries) != 1 {
		t.Fatalf("expected 1 wisdom entry, got %d", len(recorder.entries))
	}

	entry := recorder.entries[0]

	// Verify entry fields
	if entry.Key.Size != 256 {
		t.Errorf("entry.Key.Size = %d, want 256", entry.Key.Size)
	}

	if entry.Key.Precision != planner.PrecisionComplex64 {
		t.Errorf("entry.Key.Precision = %d, want %d", entry.Key.Precision, planner.PrecisionComplex64)
	}

	if entry.Key.CPUIdentifier == "" {
		t.Error("entry.Key.CPUIdentifier is empty")
	}

	if entry.Algorithm == "" {
		t.Error("entry.Algorithm is empty")
	}

	if entry.Timestamp.IsZero() {
		t.Error("entry.Timestamp is zero")
	}

	// Estimate should have a valid strategy
	if estimate.Strategy == fftypes.KernelAuto {
		t.Error("estimate.Strategy should not be fftypes.KernelAuto after measurement")
	}

	if estimate.Algorithm == "" {
		t.Error("estimate.Algorithm is empty")
	}
}

// TestMeasureAndSelect_ForcedStrategy tests that forced strategy skips measurement.
func TestMeasureAndSelect_ForcedStrategy(t *testing.T) {
	t.Parallel()

	// Use explicit features to avoid race conditions with other tests.
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}
	recorder := &mockWisdomRecorder{}

	// Force Stockham strategy
	estimate := MeasureAndSelect[complex64](
		256,
		features,
		PlannerMeasure,
		recorder,
		fftypes.KernelStockham,
	)

	// Should NOT record to wisdom when strategy is forced
	if len(recorder.entries) != 0 {
		t.Errorf("expected 0 wisdom entries with forced strategy, got %d", len(recorder.entries))
	}

	// Should use forced strategy
	if estimate.Strategy != fftypes.KernelStockham {
		t.Errorf("estimate.Strategy = %v, want %v", estimate.Strategy, fftypes.KernelStockham)
	}
}

// TestMeasureAndSelect_NilWisdom tests that MeasureAndSelect works with nil wisdom recorder.
func TestMeasureAndSelect_NilWisdom(t *testing.T) {
	t.Parallel()

	// Use explicit features to avoid race conditions with other tests.
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}

	// Should not panic with nil wisdom recorder
	estimate := MeasureAndSelect[complex64](
		256,
		features,
		PlannerMeasure,
		nil,
		fftypes.KernelAuto,
	)

	// Should still return valid estimate
	if estimate.Strategy == fftypes.KernelAuto {
		t.Error("estimate.Strategy should not be fftypes.KernelAuto")
	}
}

// TestMeasureAndSelect_Complex128 tests MeasureAndSelect with complex128.
func TestMeasureAndSelect_Complex128(t *testing.T) {
	t.Parallel()

	// Use explicit features to avoid race conditions with other tests.
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}
	recorder := &mockWisdomRecorder{}

	estimate := MeasureAndSelect[complex128](
		128,
		features,
		PlannerMeasure,
		recorder,
		fftypes.KernelAuto,
	)

	if len(recorder.entries) != 1 {
		t.Fatalf("expected 1 wisdom entry, got %d", len(recorder.entries))
	}

	// Should record with complex128 precision
	if recorder.entries[0].Key.Precision != planner.PrecisionComplex128 {
		t.Errorf("precision = %d, want %d", recorder.entries[0].Key.Precision, planner.PrecisionComplex128)
	}

	if estimate.Strategy == fftypes.KernelAuto {
		t.Error("estimate.Strategy should not be fftypes.KernelAuto")
	}
}

// TestMeasureAndSelect_AllModes tests all planner modes.
func TestMeasureAndSelect_AllModes(t *testing.T) {
	t.Parallel()

	// Use explicit features to avoid race conditions with other tests.
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}

	modes := []PlannerMode{PlannerMeasure, PlannerPatient, PlannerExhaustive}

	for _, mode := range modes {
		t.Run(mode.String(), func(t *testing.T) {
			t.Parallel()

			recorder := &mockWisdomRecorder{}
			estimate := MeasureAndSelect[complex64](
				512,
				features,
				mode,
				recorder,
				fftypes.KernelAuto,
			)

			if len(recorder.entries) != 1 {
				t.Errorf("mode %v: expected 1 entry, got %d", mode, len(recorder.entries))
			}

			if estimate.Strategy == fftypes.KernelAuto {
				t.Errorf("mode %v: strategy should not be Auto", mode)
			}
		})
	}
}

// String returns a string representation of PlannerMode for test output.
func (m PlannerMode) String() string {
	switch m {
	case PlannerEstimate:
		return "Estimate"
	case PlannerMeasure:
		return "Measure"
	case PlannerPatient:
		return "Patient"
	case PlannerExhaustive:
		return "Exhaustive"
	default:
		return "Unknown"
	}
}

// TestWisdomEntry_Timestamp tests that wisdom entries have valid timestamps.
func TestWisdomEntry_Timestamp(t *testing.T) {
	t.Parallel()

	before := time.Now()

	// Use explicit features to avoid race conditions with other tests.
	features := cpu.Features{
		ForceGeneric: true,
		Architecture: "amd64",
	}
	recorder := &mockWisdomRecorder{}

	MeasureAndSelect[complex64](64, features, PlannerMeasure, recorder, fftypes.KernelAuto)

	after := time.Now()

	if len(recorder.entries) != 1 {
		t.Fatal("expected 1 entry")
	}

	ts := recorder.entries[0].Timestamp
	if ts.Before(before) || ts.After(after) {
		t.Errorf("timestamp %v not between %v and %v", ts, before, after)
	}
}
