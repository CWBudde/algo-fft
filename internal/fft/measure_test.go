package fft

import (
	"testing"
	"time"

	"github.com/MeKo-Christian/algoforge/internal/cpu"
)

// mockWisdomRecorder records wisdom entries for testing.
type mockWisdomRecorder struct {
	entries []WisdomEntry
}

func (m *mockWisdomRecorder) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (string, bool) {
	for _, e := range m.entries {
		if e.Key.Size == size && e.Key.Precision == precision && e.Key.CPUFeatures == cpuFeatures {
			return e.Algorithm, true
		}
	}

	return "", false
}

func (m *mockWisdomRecorder) Store(entry WisdomEntry) {
	m.entries = append(m.entries, entry)
}

func TestSelectStrategiesToTest(t *testing.T) {
	tests := []struct {
		name     string
		mode     PlannerMode
		n        int
		expected []KernelStrategy
	}{
		{
			name:     "Measure mode power-of-two",
			mode:     PlannerMeasure,
			n:        1024,
			expected: []KernelStrategy{KernelDIT, KernelStockham},
		},
		{
			name:     "Patient mode power-of-two",
			mode:     PlannerPatient,
			n:        1024,
			expected: []KernelStrategy{KernelDIT, KernelStockham, KernelSixStep},
		},
		{
			name:     "Exhaustive mode power-of-two",
			mode:     PlannerExhaustive,
			n:        1024,
			expected: []KernelStrategy{KernelDIT, KernelStockham, KernelSixStep, KernelEightStep},
		},
		{
			name:     "Prime size uses Bluestein only",
			mode:     PlannerExhaustive,
			n:        17,
			expected: []KernelStrategy{KernelBluestein},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
	tests := []struct {
		mode           PlannerMode
		expectedWarmup int
		expectedIters  int
	}{
		{PlannerMeasure, 3, 10},
		{PlannerPatient, 5, 50},
		{PlannerExhaustive, 10, 100},
		{PlannerEstimate, 3, 10}, // fallback
	}

	for _, tt := range tests {
		config := getMeasureConfig(tt.mode)
		if config.warmup != tt.expectedWarmup {
			t.Errorf("getMeasureConfig(%v).warmup = %d, want %d", tt.mode, config.warmup, tt.expectedWarmup)
		}

		if config.iters != tt.expectedIters {
			t.Errorf("getMeasureConfig(%v).iters = %d, want %d", tt.mode, config.iters, tt.expectedIters)
		}
	}
}

func TestBenchmarkStrategy(t *testing.T) {
	features := cpu.DetectFeatures()

	tests := []struct {
		name     string
		n        int
		strategy KernelStrategy
	}{
		{"DIT 64", 64, KernelDIT},
		{"Stockham 64", 64, KernelStockham},
		{"DIT 256", 256, KernelDIT},
		{"Stockham 256", 256, KernelStockham},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := measureConfig{warmup: 1, iters: 3}
			elapsed := benchmarkStrategy[complex64](tt.n, features, tt.strategy, config)

			// Should complete without panicking and return positive duration
			if elapsed <= 0 {
				t.Errorf("benchmarkStrategy returned %v, expected positive duration", elapsed)
			}
		})
	}
}

func TestMeasureAndSelect_RecordsToWisdom(t *testing.T) {
	features := cpu.DetectFeatures()
	recorder := &mockWisdomRecorder{}

	// Run with PlannerMeasure mode
	estimate := MeasureAndSelect[complex64](
		256,
		features,
		PlannerMeasure,
		recorder,
		KernelAuto,
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

	if entry.Key.Precision != PrecisionComplex64 {
		t.Errorf("entry.Key.Precision = %d, want %d", entry.Key.Precision, PrecisionComplex64)
	}

	if entry.Algorithm == "" {
		t.Error("entry.Algorithm is empty")
	}

	if entry.Timestamp.IsZero() {
		t.Error("entry.Timestamp is zero")
	}

	// Estimate should have a valid strategy
	if estimate.Strategy == KernelAuto {
		t.Error("estimate.Strategy should not be KernelAuto after measurement")
	}

	if estimate.Algorithm == "" {
		t.Error("estimate.Algorithm is empty")
	}
}

func TestMeasureAndSelect_ForcedStrategy(t *testing.T) {
	features := cpu.DetectFeatures()
	recorder := &mockWisdomRecorder{}

	// Force Stockham strategy
	estimate := MeasureAndSelect[complex64](
		256,
		features,
		PlannerMeasure,
		recorder,
		KernelStockham,
	)

	// Should NOT record to wisdom when strategy is forced
	if len(recorder.entries) != 0 {
		t.Errorf("expected 0 wisdom entries with forced strategy, got %d", len(recorder.entries))
	}

	// Should use forced strategy
	if estimate.Strategy != KernelStockham {
		t.Errorf("estimate.Strategy = %v, want %v", estimate.Strategy, KernelStockham)
	}
}

func TestMeasureAndSelect_NilWisdom(t *testing.T) {
	features := cpu.DetectFeatures()

	// Should not panic with nil wisdom recorder
	estimate := MeasureAndSelect[complex64](
		256,
		features,
		PlannerMeasure,
		nil,
		KernelAuto,
	)

	// Should still return valid estimate
	if estimate.Strategy == KernelAuto {
		t.Error("estimate.Strategy should not be KernelAuto")
	}
}

func TestMeasureAndSelect_Complex128(t *testing.T) {
	features := cpu.DetectFeatures()
	recorder := &mockWisdomRecorder{}

	estimate := MeasureAndSelect[complex128](
		128,
		features,
		PlannerMeasure,
		recorder,
		KernelAuto,
	)

	if len(recorder.entries) != 1 {
		t.Fatalf("expected 1 wisdom entry, got %d", len(recorder.entries))
	}

	// Should record with complex128 precision
	if recorder.entries[0].Key.Precision != PrecisionComplex128 {
		t.Errorf("precision = %d, want %d", recorder.entries[0].Key.Precision, PrecisionComplex128)
	}

	if estimate.Strategy == KernelAuto {
		t.Error("estimate.Strategy should not be KernelAuto")
	}
}

func TestMeasureAndSelect_AllModes(t *testing.T) {
	features := cpu.DetectFeatures()

	modes := []PlannerMode{PlannerMeasure, PlannerPatient, PlannerExhaustive}

	for _, mode := range modes {
		t.Run(mode.String(), func(t *testing.T) {
			recorder := &mockWisdomRecorder{}
			estimate := MeasureAndSelect[complex64](
				512,
				features,
				mode,
				recorder,
				KernelAuto,
			)

			if len(recorder.entries) != 1 {
				t.Errorf("mode %v: expected 1 entry, got %d", mode, len(recorder.entries))
			}

			if estimate.Strategy == KernelAuto {
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

func TestWisdomEntry_Timestamp(t *testing.T) {
	before := time.Now()

	features := cpu.DetectFeatures()
	recorder := &mockWisdomRecorder{}

	MeasureAndSelect[complex64](64, features, PlannerMeasure, recorder, KernelAuto)

	after := time.Now()

	if len(recorder.entries) != 1 {
		t.Fatal("expected 1 entry")
	}

	ts := recorder.entries[0].Timestamp
	if ts.Before(before) || ts.After(after) {
		t.Errorf("timestamp %v not between %v and %v", ts, before, after)
	}
}
