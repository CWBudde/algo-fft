package cpu

import (
	"runtime"
	"testing"
	"time"
)

// isHighPrecisionPlatform returns true if the platform has a high-precision cycle counter.
// Platforms without assembly support (WebAssembly, etc.) use time.Now() fallback.
func isHighPrecisionPlatform() bool {
	// Only AMD64 and ARM64 have assembly implementations
	return runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64"
}

func TestReadCycleCounter(t *testing.T) {
	// Test that cycle counter is monotonically increasing
	c1 := ReadCycleCounter()

	// On low-precision platforms, add a small delay to ensure time progresses
	if !isHighPrecisionPlatform() {
		time.Sleep(time.Microsecond)
	}

	c2 := ReadCycleCounter()

	if c2 <= c1 {
		t.Errorf("Cycle counter not monotonic: c1=%d, c2=%d", c1, c2)
	}
}

func TestCyclesSince(t *testing.T) {
	start := ReadCycleCounter()

	// Do some work to ensure cycles elapse
	sum := 0
	for i := range 1000 {
		sum += i
	}

	// On low-precision platforms, add a delay to ensure time progresses
	if !isHighPrecisionPlatform() {
		time.Sleep(time.Microsecond)
	}

	elapsed := CyclesSince(start)

	if elapsed <= 0 {
		t.Errorf("CyclesSince returned non-positive value: %d", elapsed)
	}

	// Prevent compiler from optimizing away the loop
	if sum == 0 {
		t.Fatal("sum should not be zero")
	}
}

func TestCyclesToNanoseconds(t *testing.T) {
	// Measure a known duration and verify cycle-to-ns conversion is reasonable
	start := ReadCycleCounter()
	timeStart := time.Now()

	// Sleep for a measurable duration
	time.Sleep(10 * time.Millisecond)

	cycles := CyclesSince(start)
	actualNanos := time.Since(timeStart).Nanoseconds()
	convertedNanos := CyclesToNanoseconds(cycles)

	// Conversion should be within 50% of actual time
	// (loose tolerance due to calibration, sleep precision, and scheduler noise)
	ratio := float64(convertedNanos) / float64(actualNanos)
	if ratio < 0.5 || ratio > 2.0 {
		t.Errorf("Cycle-to-nanosecond conversion appears incorrect: got %d ns from cycles, actual %d ns (ratio %.2f)",
			convertedNanos, actualNanos, ratio)
	}
}

func TestCycleCounterPrecision(t *testing.T) {
	// Skip this test on low-precision platforms where time.Now() is used
	if !isHighPrecisionPlatform() {
		t.Skip("Skipping precision test on platform without hardware cycle counter")
	}

	// Measure how many unique values we can read in rapid succession
	const samples = 1000

	values := make([]int64, samples)

	for i := range values {
		values[i] = ReadCycleCounter()
	}

	// Count unique values
	unique := make(map[int64]bool)
	for _, v := range values {
		unique[v] = true
	}

	// On real cycle counters, we should get many unique values
	// Require at least 10% uniqueness (very conservative)
	uniqueRatio := float64(len(unique)) / float64(samples)
	if uniqueRatio < 0.1 {
		t.Errorf("Cycle counter has low precision: only %.1f%% unique values in %d samples",
			uniqueRatio*100, samples)
	}

	t.Logf("Cycle counter uniqueness: %.1f%% (%d unique values in %d samples)",
		uniqueRatio*100, len(unique), samples)
}

func BenchmarkReadCycleCounter(b *testing.B) {
	for range b.N {
		_ = ReadCycleCounter()
	}
}

func BenchmarkCyclesSince(b *testing.B) {
	start := ReadCycleCounter()
	for range b.N {
		_ = CyclesSince(start)
	}
}

func BenchmarkCyclesToNanoseconds(b *testing.B) {
	cycles := int64(1000000)
	for range b.N {
		_ = CyclesToNanoseconds(cycles)
	}
}
